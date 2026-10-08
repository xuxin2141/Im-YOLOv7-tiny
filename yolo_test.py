import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm
import pandas as pd

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel


# 新增：数据保存函数
def save_detailed_metrics(save_dir, p, r, ap, f1, ap_class, names,
                          pr_curve_data, f1_curve_data, precision_curve_data, recall_curve_data,
                          seen, nt, mp, mr, map50, map, t0, t1, prefix='val'):
    """保存详细的评估指标数据"""
    save_dir = Path(save_dir) / "detailed_metrics"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n保存详细指标数据到: {save_dir}")

    # 1. 保存每个类别的指标
    class_metrics = []
    for i, c in enumerate(ap_class):
        class_name = names[c] if c < len(names) else f'class_{c}'

        # 修复：确保所有值都是标量
        precision_val = float(p[i]) if not isinstance(p[i], (np.ndarray, list)) else float(p[i][0])
        recall_val = float(r[i]) if not isinstance(r[i], (np.ndarray, list)) else float(r[i][0])
        f1_val = float(f1[i]) if not isinstance(f1[i], (np.ndarray, list)) else float(f1[i][0])

        # 修复：根据ap的维度处理数据
        if isinstance(ap, np.ndarray) and ap.ndim == 2:  # ap是二维数组
            ap50_val = float(ap[i, 0])
            ap_mean_val = float(ap[i].mean())
            ap_array = ap[i].tolist()
        else:  # ap是一维数组
            ap_val = ap[i] if not isinstance(ap[i], (np.ndarray, list)) else ap[i][0]
            ap50_val = float(ap_val)
            ap_mean_val = float(ap_val)
            ap_array = [ap_val]

        class_metrics.append({
            'class_id': int(c),
            'class_name': class_name,
            'precision': precision_val,
            'recall': recall_val,
            'f1_score': f1_val,
            'ap50': ap50_val,  # AP@0.5
            'ap': ap_mean_val,  # AP@0.5:0.95
            'ap_array': ap_array  # 所有IoU阈值下的AP
        })

    # 保存类别指标
    df_class = pd.DataFrame(class_metrics)
    df_class.to_csv(save_dir / f'{prefix}_class_metrics.csv', index=False)
    with open(save_dir / f'{prefix}_class_metrics.json', 'w') as f:
        json.dump(class_metrics, f, indent=2)

    # 2. 保存PR曲线数据
    if pr_curve_data:
        # 保存每个类别的PR曲线
        for class_name, data in pr_curve_data.items():
            # 确保数据是列表格式
            recall_data = data['recall'] if isinstance(data['recall'], list) else data['recall'].tolist()
            precision_data = data['precision'] if isinstance(data['precision'], list) else data['precision'].tolist()

            df_pr = pd.DataFrame({
                'recall': recall_data,
                'precision': precision_data
            })
            # 使用安全的文件名
            safe_class_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            df_pr.to_csv(save_dir / f'{prefix}_pr_curve_{safe_class_name}.csv', index=False)

        # 保存汇总PR数据
        with open(save_dir / f'{prefix}_pr_curves.json', 'w') as f:
            json.dump(pr_curve_data, f, indent=2)

    # 3. 保存F1曲线数据
    if f1_curve_data:
        # 保存每个类别的F1曲线
        for class_name, data in f1_curve_data.items():
            # 确保数据是列表格式
            thresholds = data['confidence_thresholds'] if isinstance(data['confidence_thresholds'], list) else data[
                'confidence_thresholds'].tolist()
            f1_scores = data['f1_scores'] if isinstance(data['f1_scores'], list) else data['f1_scores'].tolist()

            df_f1 = pd.DataFrame({
                'confidence_threshold': thresholds,
                'f1_score': f1_scores
            })
            # 使用安全的文件名
            safe_class_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            df_f1.to_csv(save_dir / f'{prefix}_f1_curve_{safe_class_name}.csv', index=False)

        # 保存汇总F1数据
        with open(save_dir / f'{prefix}_f1_curves.json', 'w') as f:
            json.dump(f1_curve_data, f, indent=2)

    # 4. 保存精确率曲线数据
    if precision_curve_data:
        for class_name, data in precision_curve_data.items():
            # 确保数据是列表格式
            thresholds = data['confidence_thresholds'] if isinstance(data['confidence_thresholds'], list) else data[
                'confidence_thresholds'].tolist()
            precision_scores = data['precision_scores'] if isinstance(data['precision_scores'], list) else data[
                'precision_scores'].tolist()

            df_p = pd.DataFrame({
                'confidence_threshold': thresholds,
                'precision': precision_scores
            })
            # 使用安全的文件名
            safe_class_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            df_p.to_csv(save_dir / f'{prefix}_precision_curve_{safe_class_name}.csv', index=False)

        with open(save_dir / f'{prefix}_precision_curves.json', 'w') as f:
            json.dump(precision_curve_data, f, indent=2)

    # 5. 保存召回率曲线数据
    if recall_curve_data:
        for class_name, data in recall_curve_data.items():
            # 确保数据是列表格式
            thresholds = data['confidence_thresholds'] if isinstance(data['confidence_thresholds'], list) else data[
                'confidence_thresholds'].tolist()
            recall_scores = data['recall_scores'] if isinstance(data['recall_scores'], list) else data[
                'recall_scores'].tolist()

            df_r = pd.DataFrame({
                'confidence_threshold': thresholds,
                'recall': recall_scores
            })
            # 使用安全的文件名
            safe_class_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            df_r.to_csv(save_dir / f'{prefix}_recall_curve_{safe_class_name}.csv', index=False)

        with open(save_dir / f'{prefix}_recall_curves.json', 'w') as f:
            json.dump(recall_curve_data, f, indent=2)

    # 6. 保存总体指标
    overall_metrics = {
        'total_images': int(seen),
        'total_targets': int(nt.sum()),
        'mean_precision': float(mp),
        'mean_recall': float(mr),
        'mean_f1': float(np.mean(f1)),
        'map50': float(map50),
        'map': float(map),
        'inference_time_ms': float(t0 / seen * 1E3),
        'nms_time_ms': float(t1 / seen * 1E3),
        'total_time_ms': float((t0 + t1) / seen * 1E3)
    }

    with open(save_dir / f'{prefix}_overall_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=2)

    df_overall = pd.DataFrame([overall_metrics])
    df_overall.to_csv(save_dir / f'{prefix}_overall_metrics.csv', index=False)

    print(f"已保存 {len(class_metrics)} 个类别的详细指标数据")
    print(f"数据位置: {save_dir}")


def cs(data,
       weights=None,
       batch_size=32,
       imgsz=640,
       conf_thres=0.001,
       iou_thres=0.6,  # for NMS
       save_json=False,
       single_cls=False,
       augment=False,
       verbose=False,
       model=None,
       dataloader=None,
       save_dir=Path(''),  # for saving images
       save_txt=False,  # for auto-labelling
       save_hybrid=False,  # for hybrid auto-labelling
       save_conf=False,  # save auto-label confidences
       plots=True,
       wandb_logger=None,
       compute_loss=None,
       half_precision=True,
       trace=False,
       is_coco=False,
       v5_metric=False):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        if trace:
            model = TracedModel(model, device, imgsz)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        try:
            # 修改这里：调用修改后的ap_per_class函数，获取曲线数据
            p, r, ap, f1, ap_class, pr_curve_data, f1_curve_data, precision_curve_data, recall_curve_data = ap_per_class(
                *stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names, return_curves_data=True
            )

            # 修复：确保所有值都是标量
            p = np.array([float(x) if not isinstance(x, (np.ndarray, list)) else float(x[0]) for x in p])
            r = np.array([float(x) if not isinstance(x, (np.ndarray, list)) else float(x[0]) for x in r])
            f1 = np.array([float(x) if not isinstance(x, (np.ndarray, list)) else float(x[0]) for x in f1])

            # 修复：根据ap的维度计算ap50和map
            if isinstance(ap, np.ndarray) and ap.ndim == 2:  # ap是二维数组
                ap50 = np.array([float(ap[i, 0]) for i in range(len(ap))])
                map50 = float(ap50.mean())
                map_val = float(ap.mean())
            else:  # ap是一维数组
                ap50 = np.array([float(ap[i]) if not isinstance(ap[i], (np.ndarray, list)) else float(ap[i][0]) for i in
                                 range(len(ap))])
                map50 = float(ap50.mean())
                map_val = float(ap.mean()) if not isinstance(ap, (np.ndarray, list)) else float(np.mean(ap))

            mp, mr = float(p.mean()), float(r.mean())
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class

            # 新增：保存详细指标数据
            save_detailed_metrics(
                save_dir, p, r, ap, f1, ap_class, names,
                pr_curve_data, f1_curve_data, precision_curve_data, recall_curve_data,
                seen, nt, mp, mr, map50, map_val, t0, t1
            )

        except Exception as e:
            print(f"计算指标时出错: {e}")
            # 如果出错，使用原始方法计算
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir,
                                                  names=names)

            # 修复：确保所有值都是标量
            p = np.array([float(x) if not isinstance(x, (np.ndarray, list)) else float(x[0]) for x in p])
            r = np.array([float(x) if not isinstance(x, (np.ndarray, list)) else float(x[0]) for x in r])
            f1 = np.array([float(x) if not isinstance(x, (np.ndarray, list)) else float(x[0]) for x in f1])

            if isinstance(ap, np.ndarray) and ap.ndim == 2:
                ap50 = np.array([float(ap[i, 0]) for i in range(len(ap))])
                map50 = float(ap50.mean())
                map_val = float(ap.mean())
            else:
                ap50 = np.array([float(ap[i]) if not isinstance(ap[i], (np.ndarray, list)) else float(ap[i][0]) for i in
                                 range(len(ap))])
                map50 = float(ap50.mean())
                map_val = float(ap.mean()) if not isinstance(ap, (np.ndarray, list)) else float(np.mean(ap))

            mp, mr = float(p.mean()), float(r.mean())
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format

    # 修复：确保打印的值都是标量
    print(pf % ('all', seen, int(nt.sum()), mp, mr, map50, map_val))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            # 修复：确保打印的值都是标量
            p_val = float(p[i]) if not isinstance(p[i], (np.ndarray, list)) else float(p[i][0])
            r_val = float(r[i]) if not isinstance(r[i], (np.ndarray, list)) else float(r[i][0])
            ap50_val = float(ap50[i]) if not isinstance(ap50[i], (np.ndarray, list)) else float(ap50[i][0])
            ap_val = float(ap[i]) if not isinstance(ap[i], (np.ndarray, list)) else float(ap[i][0])

            print(pf % (names[c], seen, int(nt[c]), p_val, r_val, ap50_val, ap_val))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    # 修复：确保返回值都是标量
    if isinstance(ap, np.ndarray) and ap.ndim == 2:
        maps = np.zeros(nc) + map_val
        for i, c in enumerate(ap_class):
            maps[c] = float(ap[i].mean())
    else:
        maps = np.zeros(nc) + map_val
        for i, c in enumerate(ap_class):
            maps[c] = float(ap[i]) if not isinstance(ap[i], (np.ndarray, list)) else float(ap[i][0])

    return (mp, mr, map50, map_val, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='yolo_test.py')
    parser.add_argument('--weights', nargs='+', type=str,
                        default='D:/YOLO/yolov7-main/runs/train/yolov7-yingguang/weights/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='D:/YOLO/yolov7-main/data/mydata.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='SimAM', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    # check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        cs(opt.data,
           opt.weights,
           opt.batch_size,
           opt.img_size,
           opt.conf_thres,
           opt.iou_thres,
           opt.save_json,
           opt.single_cls,
           opt.augment,
           opt.verbose,
           save_txt=opt.save_txt | opt.save_hybrid,
           save_hybrid=opt.save_hybrid,
           save_conf=opt.save_conf,
           trace=not opt.no_trace,
           v5_metric=opt.v5_metric
           )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            cs(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False,
               v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python yolo_test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = cs(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                             plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot