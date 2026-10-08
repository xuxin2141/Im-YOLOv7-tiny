# -*- coding=utf-8 -*-

import time
import random
import copy
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
import argparse


# 显示图片
def show_pic(img, bboxes=None, labels=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        labels:每个box对应的类别
    '''
    img_display = img.copy()
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img_display, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 3)
        if labels is not None:
            cv2.putText(img_display, str(labels[i]), (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # 修复了这行的括号
    cv2.namedWindow('pic', 0)
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200, 800)
    cv2.imshow('pic', img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 图像均为cv2读取
class DataAugmentForObjectDetection():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=5,
                 crop_rate=0.5, shift_rate=0.5, change_light_rate=0.5,
                 add_noise_rate=0.5, flip_rate=0.5,
                 cutout_rate=0.5, cut_out_length=50, cut_out_holes=1, cut_out_threshold=0.5,
                 is_addNoise=True, is_changeLight=True, is_cutout=True, is_rotate_img_bbox=True,
                 is_crop_img_bboxes=True, is_shift_pic_bboxes=True, is_filp_pic_bboxes=True):

        # 配置各个操作的属性
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold

        # 是否使用某种增强方式
        self.is_addNoise = is_addNoise
        self.is_changeLight = is_changeLight
        self.is_cutout = is_cutout
        self.is_rotate_img_bbox = is_rotate_img_bbox
        self.is_crop_img_bboxes = is_crop_img_bboxes
        self.is_shift_pic_bboxes = is_shift_pic_bboxes
        self.is_filp_pic_bboxes = is_filp_pic_bboxes

    # ----1.加噪声---- #
    def _addNoise(self, img):
        return random_noise(img, mode='gaussian', seed=int(time.time()), clip=True) * 255

    # ---2.调整亮度--- #
    def _changeLight(self, img):
        alpha = random.uniform(0.35, 1)
        blank = np.zeros(img.shape, img.dtype)
        return cv2.addWeighted(img, alpha, blank, 1 - alpha, 0)

    # ---3.cutout--- #
    def _cutout(self, img, bboxes, length=100, n_holes=1, threshold=0.5):
        def cal_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            interArea = (xB - xA + 1) * (yB - yA + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            iou = interArea / float(boxBArea)
            return iou

        if img.ndim == 3:
            h, w, c = img.shape
        else:
            _, h, w, c = img.shape
        mask = np.ones((h, w, c), np.float32)
        for n in range(n_holes):
            chongdie = True
            while chongdie:
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - length // 2, 0, h)
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                chongdie = False
                for box in bboxes:
                    if cal_iou([x1, y1, x2, y2], box) > threshold:
                        chongdie = True
                        break
            mask[y1: y2, x1: x2, :] = 0.
        img = img * mask
        return img

    # ---4.旋转--- #
    def _rotate_img_bbox(self, img, bboxes, angle=5, scale=1.):
        w = img.shape[1]
        h = img.shape[0]
        rangle = np.deg2rad(angle)
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
            concat = np.vstack((point1, point2, point3, point4))
            concat = concat.astype(np.int32)
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx + rw
            ry_max = ry + rh
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

        return rot_img, rot_bboxes

    # ---5.裁剪--- #
    def _crop_img_bboxes(self, img, bboxes):
        w = img.shape[1]
        h = img.shape[0]
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max

        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2] - crop_x_min, bbox[3] - crop_y_min])

        return crop_img, crop_bboxes

    # ---6.平移--- #
    def _shift_pic_bboxes(self, img, bboxes):
        w = img.shape[1]
        h = img.shape[0]
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max

        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

        M = np.float32([[1, 0, x], [0, 1, y]])
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        shift_bboxes = list()
        for bbox in bboxes:
            shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y])

        return shift_img, shift_bboxes

    # ---7.镜像--- #
    def _filp_pic_bboxes(self, img, bboxes):
        flip_img = copy.deepcopy(img)
        h, w, _ = img.shape

        sed = random.random()

        if 0 < sed < 0.33:
            flip_img = cv2.flip(flip_img, 0)
            inver = 0
        elif 0.33 < sed < 0.66:
            flip_img = cv2.flip(flip_img, 1)
            inver = 1
        else:
            flip_img = cv2.flip(flip_img, -1)
            inver = -1

        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]

            if inver == 0:
                flip_bboxes.append([x_min, h - y_max, x_max, h - y_min])
            elif inver == 1:
                flip_bboxes.append([w - x_max, y_min, w - x_min, y_max])
            elif inver == -1:
                flip_bboxes.append([w - x_max, h - y_max, w - x_min, h - y_min])
        return flip_img, flip_bboxes

    # 图像增强方法
    def dataAugment(self, img, bboxes):
        change_num = 0
        while change_num < 1:
            if self.is_rotate_img_bbox:
                if random.random() > self.rotation_rate:
                    change_num += 1
                    angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                    scale = random.uniform(0.7, 0.8)
                    img, bboxes = self._rotate_img_bbox(img, bboxes, angle, scale)

            if self.is_shift_pic_bboxes:
                if random.random() < self.shift_rate:
                    change_num += 1
                    img, bboxes = self._shift_pic_bboxes(img, bboxes)

            if self.is_changeLight:
                if random.random() > self.change_light_rate:
                    change_num += 1
                    img = self._changeLight(img)

            if self.is_addNoise:
                if random.random() < self.add_noise_rate:
                    change_num += 1
                    img = self._addNoise(img)
            if self.is_cutout:
                if random.random() < self.cutout_rate:
                    change_num += 1
                    img = self._cutout(img, bboxes, length=self.cut_out_length, n_holes=self.cut_out_holes,
                                       threshold=self.cut_out_threshold)
            if self.is_filp_pic_bboxes:
                if random.random() < self.flip_rate:
                    change_num += 1
                    img, bboxes = self._filp_pic_bboxes(img, bboxes)

        return img, bboxes


# TXT标注文件处理工具
class TxtToolHelper():
    # 从txt文件中提取bounding box信息
    def parse_txt(self, path, img_width, img_height):
        '''
        输入:
            path: txt文件路径
            img_width: 图像宽度
            img_height: 图像高度
        输出:
            从txt文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, class_id]]
        '''
        coords = list()
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split()
                if len(data) < 5:
                    continue
                class_id = int(data[0])
                x_center = float(data[1])
                y_center = float(data[2])
                width = float(data[3])
                height = float(data[4])

                # 转换为绝对坐标
                x_min = (x_center - width / 2) * img_width
                y_min = (y_center - height / 2) * img_height
                x_max = (x_center + width / 2) * img_width
                y_max = (y_center + height / 2) * img_height

                coords.append([x_min, y_min, x_max, y_max, class_id])
        return coords

    # 保存图片结果
    def save_img(self, file_name, save_folder, img):
        cv2.imwrite(os.path.join(save_folder, file_name), img)

    # 保存txt结果
    def save_txt(self, file_name, save_folder, img_info, height, width, bboxs_info):
        '''
        :param file_name: 文件名
        :param save_folder: 保存的txt文件路径
        :param img_info: 图片信息（暂时不用）
        :param height: 图片高度
        :param width: 图片宽度
        :param bboxs_info: 边框和标签信息 (labels, bboxs)
        '''
        labels, bboxs = bboxs_info

        with open(os.path.join(save_folder, file_name), 'w') as f:
            for label, box in zip(labels, bboxs):
                x_min, y_min, x_max, y_max = box

                # 转换为YOLO格式 (归一化坐标)
                x_center = (x_min + x_max) / (2 * width)
                y_center = (y_min + y_max) / (2 * height)
                bbox_width = (x_max - x_min) / width
                bbox_height = (y_max - y_min) / height

                # 确保坐标在[0,1]范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                bbox_width = max(0, min(1, bbox_width))
                bbox_height = max(0, min(1, bbox_height))

                f.write(f"{label} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")


if __name__ == '__main__':
    need_aug_num = 20  # 每张图片需要增强的次数，改为20倍

    is_endwidth_dot = True

    dataAug = DataAugmentForObjectDetection()  # 数据增强工具类

    toolhelper = TxtToolHelper()  # TXT工具

    # 获取相关参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_img_path', type=str,
                        default=r'D:\Desk\yingguang\yingguang\zengqiangim\qian',
                        help='源图片路径')
    parser.add_argument('--source_txt_path', type=str,
                        default=r'D:\Desk\yingguang\yingguang\zengqianglabel\qian',
                        help='源TXT标注文件路径')
    parser.add_argument('--save_img_path', type=str,
                        default=r'D:\Desk\yingguang\yingguang\zengqiangim\hou',
                        help='增强后图片保存路径')
    parser.add_argument('--save_txt_path', type=str,
                        default=r'D:\Desk\yingguang\yingguang\zengqianglabel\hou',
                        help='增强后TXT标注保存路径')
    parser.add_argument('--show_image', type=bool, default=False, help='是否显示增强后的图像')
    args = parser.parse_args()

    source_img_path = args.source_img_path
    source_txt_path = args.source_txt_path
    save_img_path = args.save_img_path
    save_txt_path = args.save_txt_path
    show_image = args.show_image

    # 如果保存文件夹不存在就创建
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    if not os.path.exists(save_txt_path):
        os.makedirs(save_txt_path)

    # 获取所有图片文件
    img_files = [f for f in os.listdir(source_img_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"找到 {len(img_files)} 张图片，开始数据增强...")

    for img_file in img_files:
        cnt = 0
        pic_path = os.path.join(source_img_path, img_file)

        # 对应的TXT文件路径
        txt_file = os.path.splitext(img_file)[0] + '.txt'
        txt_path = os.path.join(source_txt_path, txt_file)

        if not os.path.exists(txt_path):
            print(f"警告: 找不到对应的标注文件 {txt_path}，跳过图片 {img_file}")
            continue

        # 读取图片获取尺寸
        img = cv2.imread(pic_path)
        if img is None:
            print(f"错误: 无法读取图片 {pic_path}")
            continue

        img_height, img_width = img.shape[:2]

        # 解析TXT标注
        values = toolhelper.parse_txt(txt_path, img_width, img_height)
        coords = [v[:4] for v in values]  # 得到框坐标
        labels = [v[-1] for v in values]  # 得到类别标签

        print(f"处理图片: {img_file}, 找到 {len(coords)} 个标注框")

        if is_endwidth_dot:
            dot_index = img_file.rfind('.')
            _file_prefix = img_file[:dot_index]
            _file_suffix = img_file[dot_index:]

        # 开始增强
        while cnt < need_aug_num:
            try:
                auged_img, auged_bboxes = dataAug.dataAugment(img.copy(), copy.deepcopy(coords))
                auged_bboxes_int = np.array(auged_bboxes).astype(np.int32)
                height, width, channel = auged_img.shape

                # 生成文件名
                img_name = '{}_{}{}'.format(_file_prefix, cnt + 1, _file_suffix)
                txt_name = '{}_{}.txt'.format(_file_prefix, cnt + 1)

                # 保存增强后的图片和标注
                toolhelper.save_img(img_name, save_img_path, auged_img)
                toolhelper.save_txt(txt_name, save_txt_path, (save_img_path, img_name),
                                    height, width, (labels, auged_bboxes_int))

                print(f"生成增强数据: {img_name}")
                cnt += 1

            except Exception as e:
                print(f"增强过程中出现错误: {e}")
                cnt += 1
                continue

    print("数据增强完成！")