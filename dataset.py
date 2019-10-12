# coding=utf-8
'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version: 
@Author: Aoru Xue
@Date: 2019-09-10 12:51:20
@LastEditors: Aoru Xue
@LastEditTime: 2019-10-01 10:33:38
'''
import torch
import torch._C
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import dataloader
import glob
import pickle
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np  
from priors import Priors
from basket_utils import *
from torch._six import int_classes as _int_classes
import cv2
from augment.augmentor import Augmentor
import random

np.set_printoptions(threshold=np.inf)


class LFFDDatasetBase(Dataset):
    def __init__(self,
                 enable_horizon_flip,
                 enable_vertical_flip,
                 enable_random_brightness,
                 brightness_params,
                 enable_random_saturation,
                 saturation_params,
                 enable_random_contrast,
                 contrast_params,
                 enable_blur,
                 blur_params,
                 blur_kernel_size_list,
                 num_image_channels,
                 net_input_height,
                 net_input_width,
                 num_output_scales,
                 receptive_field_list,
                 receptive_field_stride,
                 feature_map_size_list,
                 receptive_field_center_start,
                 bbox_small_list,
                 bbox_large_list,
                 bbox_small_gray_list,
                 bbox_large_gray_list,
                 num_output_channels,
                 neg_image_resize_factor_interval
                 ):

        # augmentation settings
        self.enable_horizon_flip = enable_horizon_flip
        self.enable_vertical_flip = enable_vertical_flip
        self.pixel_augmentor_func_list = []

        self.enable_random_brightness = enable_random_brightness
        self.brightness_params = brightness_params
        def brightness_augmentor(input_im):
            if self.enable_random_brightness and random.random() > 0.5:
                input_im = Augmentor.brightness(input_im, **self.brightness_params)
                return input_im
            else:
                return input_im
        self.pixel_augmentor_func_list.append(brightness_augmentor)

        self.enable_random_saturation = enable_random_saturation
        self.saturation_params = saturation_params
        def saturation_augmentor(input_im):
            if self.enable_random_saturation and random.random() > 0.5:
                input_im = Augmentor.saturation(input_im, **self.saturation_params)
                return input_im
            else:
                return input_im
        self.pixel_augmentor_func_list.append(saturation_augmentor)

        self.enable_random_contrast = enable_random_contrast
        self.contrast_params = contrast_params
        def contrast_augmentor(input_im):
            if self.enable_random_contrast and random.random() > 0.5:
                input_im = Augmentor.contrast(input_im, **self.contrast_params)
                return input_im
            else:
                return input_im
        self.pixel_augmentor_func_list.append(contrast_augmentor)

        self.enable_blur = enable_blur
        self.blur_params = blur_params
        self.blur_kernel_size_list = blur_kernel_size_list
        def blur_augmentor(input_im):
            if self.enable_blur and random.random() > 0.5:
                kernel_size = random.choice(self.blur_kernel_size_list)
                self.blur_params['kernel_size'] = kernel_size
                input_im = Augmentor.blur(input_im, **self.blur_params)
                return input_im
            else:
                return input_im
        self.pixel_augmentor_func_list.append(blur_augmentor)

        self.num_image_channels = num_image_channels
        self.net_input_height = net_input_height
        self.net_input_width = net_input_width

        self.num_output_scales = num_output_scales
        self.receptive_field_list = receptive_field_list
        self.feature_map_size_list = feature_map_size_list
        self.receptive_field_stride = receptive_field_stride
        self.bbox_small_list = bbox_small_list
        self.bbox_large_list = bbox_large_list
        self.receptive_field_center_start = receptive_field_center_start
        self.normalization_constant = [i / 2.0 for i in self.receptive_field_list]
        self.bbox_small_gray_list = bbox_small_gray_list
        self.bbox_large_gray_list = bbox_large_gray_list
        self.num_output_channels = num_output_channels
        self.neg_image_resize_factor_interval = neg_image_resize_factor_interval

    def __getitem__(self, item):
        """read the image and construct the label and mask map for all detect head branch"""
        im = self._get_image(item)
        label_batch_list = [np.zeros((self.num_output_channels,
                                         v,
                                         v),
                                        dtype=np.float32)
                            for v in self.feature_map_size_list]

        mask_batch_list = [np.zeros((self.num_output_channels,
                                        v,
                                        v),
                                       dtype=np.float32)
                           for v in self.feature_map_size_list]
        bboxes_inner, is_pos = self._get_annotation(item)
        if is_pos is not None:
            """posiitve sample"""
            bboxes_org = np.array(bboxes_inner)


            num_bboxes = bboxes_org.shape[0]

            bboxes = bboxes_org.copy()

            # #data augmentation
            if self.enable_horizon_flip and random.random() > 0.5:
                im = Augmentor.flip(im, 'h')
                bboxes[:, 0] = im.shape[1] - (bboxes[:, 0] + bboxes[:, 2])
            if self.enable_vertical_flip and random.random() > 0.5:
                im = Augmentor.flip(im, 'v')
                bboxes[:, 1] = im.shape[0] - (bboxes[:, 1] + bboxes[:, 3])

            # randomly select a bbox
            bbox_idx = random.randint(0, num_bboxes - 1)
            # bbox_idx = 0

            # randomly select a reasonable scale for the selected bbox (selection strategy may vary from task to task)
            target_bbox = bboxes[bbox_idx, :]
            longer_side = max(target_bbox[2:])
            if longer_side <= self.bbox_small_list[0]:
                scale_idx = 0
            elif longer_side <= self.bbox_small_list[1]:
                scale_idx = random.randint(0, 1)
            elif longer_side <= self.bbox_small_list[2]:
                scale_idx = random.randint(0, 2)
            else:
                if random.random() > 0.9:
                    scale_idx = random.randint(0, self.num_output_scales)
                else:
                    scale_idx = random.randint(0, self.num_output_scales - 1)

            # choose a side length in the selected scale
            if scale_idx == self.num_output_scales:
                scale_idx -= 1
                side_length = self.bbox_large_list[-1] + random.randint(0, self.bbox_large_list[-1] * 0.5)
            else:
                side_length = self.bbox_small_list[scale_idx] + \
                              random.randint(0, self.bbox_large_list[scale_idx] - self.bbox_small_list[scale_idx])

            target_scale = float(side_length) / longer_side

            # resize bboxes
            bboxes = bboxes * target_scale
            target_bbox = target_bbox * target_scale

            # determine the states of a bbox in each scale
            green = [[False for i in range(num_bboxes)] for j in range(self.num_output_scales)]
            gray = [[False for i in range(num_bboxes)] for j in range(self.num_output_scales)]
            valid = [[False for i in range(num_bboxes)] for j in range(self.num_output_scales)]
            #根据box大小将其分配到不同的head上去预测
            for i in range(num_bboxes):
                temp_bbox = bboxes[i, :]
                large_side = max(temp_bbox[2:])
                for j in range(self.num_output_scales):
                    if self.bbox_small_list[j] <= large_side <= self.bbox_large_list[j]:
                        green[j][i] = True
                        valid[j][i] = True
                    elif self.bbox_small_gray_list[j] <= large_side <= self.bbox_large_gray_list[j]:
                        gray[j][i] = True
                        valid[j][i] = True

            # resize the original image
            im = cv2.resize(im, None, fx=target_scale, fy=target_scale)
            # crop the original image centered on the center of the selected bbox with vibration (it can be regarded as an augmentation)
            vibration_length = int(self.receptive_field_stride[scale_idx] / 2)
            # vibration_length = 0
            offset_x = random.randint(-vibration_length, vibration_length)
            offset_y = random.randint(-vibration_length, vibration_length)
            crop_left = int(target_bbox[0] + target_bbox[2] / 2 + offset_x - self.net_input_width / 2.0)
            if crop_left < 0:
                crop_left_pad = -int(crop_left)
                crop_left = 0
            else:
                crop_left_pad = 0
            crop_top = int(target_bbox[1] + target_bbox[3] / 2 + offset_y - self.net_input_height / 2.0)
            if crop_top < 0:
                crop_top_pad = -int(crop_top)
                crop_top = 0
            else:
                crop_top_pad = 0
            crop_right = int(target_bbox[0] + target_bbox[2] / 2 + offset_x + self.net_input_width / 2.0)
            if crop_right > im.shape[1]:
                crop_right = im.shape[1]

            crop_bottom = int(target_bbox[1] + target_bbox[3] / 2 + offset_y + self.net_input_height / 2.0)
            if crop_bottom > im.shape[0]:
                crop_bottom = im.shape[0]

            # while crop_bottom - crop_top >640:
            #     crop_bottom -=1
            # while crop_right - crop_left > 640:
            #     crop_right -=1
            im = im[crop_top:crop_bottom, crop_left:crop_right, :]
            im_input = np.zeros((self.net_input_height, self.net_input_width, 3), dtype=np.uint8)
            im_input[crop_top_pad:crop_top_pad + im.shape[0], crop_left_pad:crop_left_pad + im.shape[1], :] = im

            # image augmentation
            if random.random() > 0.5:
                random.shuffle(self.pixel_augmentor_func_list)
                for augmentor in self.pixel_augmentor_func_list:
                    im_input = augmentor(im_input)

            im_input = im_input.astype(dtype=np.float32)
            im_input = im_input.transpose([2, 0, 1])

            # construct GT feature maps for each scale
            label_list = []
            mask_list = []
            for i in range(self.num_output_scales):

                # compute the center coordinates of all RFs
                receptive_field_centers = np.array(
                    [self.receptive_field_center_start[i] + w * self.receptive_field_stride[i] for w in
                     range(self.feature_map_size_list[i])])

                shift_x = (self.net_input_width / 2.0 - target_bbox[2] / 2) - target_bbox[0] - offset_x
                shift_y = (self.net_input_height / 2.0 - target_bbox[3] / 2) - target_bbox[1] - offset_y
                temp_label = np.zeros(
                    (self.num_output_channels, self.feature_map_size_list[i], self.feature_map_size_list[i]),
                    dtype=np.float32)
                temp_mask = np.zeros(
                    (self.num_output_channels, self.feature_map_size_list[i], self.feature_map_size_list[i]),
                    dtype=np.float32)
                temp_label[1, :, :] = 1
                temp_mask[0:2, :, :] = 1

                score_map_green = np.zeros((self.feature_map_size_list[i], self.feature_map_size_list[i]),
                                              dtype=np.int32)
                score_map_gray = np.zeros((self.feature_map_size_list[i], self.feature_map_size_list[i]),
                                             dtype=np.int32)
                for j in range(num_bboxes):
                    if not valid[i][j]:
                        continue
                    temp_bbox = bboxes[j, :]

                    # skip the bbox that does not appear in the cropped area
                    if temp_bbox[0] + temp_bbox[2] + shift_x <= 0 or temp_bbox[0] + shift_x >= self.net_input_width \
                            or temp_bbox[1] + temp_bbox[3] + shift_y <= 0 or temp_bbox[
                        1] + shift_y >= self.net_input_height:
                        continue

                    temp_bbox_left_bound = temp_bbox[0] + shift_x
                    temp_bbox_right_bound = temp_bbox[0] + temp_bbox[2] + shift_x
                    temp_bbox_top_bound = temp_bbox[1] + shift_y
                    temp_bbox_bottom_bound = temp_bbox[1] + temp_bbox[3] + shift_y

                    left_rf_center_index = int(max(0, math.ceil((temp_bbox_left_bound - self.receptive_field_center_start[i]) / self.receptive_field_stride[i])))

                    right_rf_center_index = int(min(self.feature_map_size_list[i] - 1, math.floor((temp_bbox_right_bound - self.receptive_field_center_start[i]) / self.receptive_field_stride[i])))

                    top_rf_center_index = int(max(0, math.ceil((temp_bbox_top_bound - self.receptive_field_center_start[i]) / self.receptive_field_stride[i])))

                    bottom_rf_center_index = int(min(self.feature_map_size_list[i] - 1, math.floor((temp_bbox_bottom_bound - self.receptive_field_center_start[i]) / self.receptive_field_stride[i])))

                    # ignore the face with no RF centers inside
                    if right_rf_center_index < left_rf_center_index or bottom_rf_center_index < top_rf_center_index:
                        continue

                    if gray[i][j]:
                        score_map_gray[top_rf_center_index:bottom_rf_center_index + 1, left_rf_center_index:right_rf_center_index + 1] = 1

                    else:
                        score_map_green[top_rf_center_index:bottom_rf_center_index + 1, left_rf_center_index:right_rf_center_index + 1] += 1

                        x_centers = receptive_field_centers[left_rf_center_index:right_rf_center_index + 1]
                        y_centers = receptive_field_centers[top_rf_center_index:bottom_rf_center_index + 1]
                        x0_location_regression = (x_centers - temp_bbox_left_bound) / self.normalization_constant[i]
                        y0_location_regression = (y_centers - temp_bbox_top_bound) / self.normalization_constant[i]
                        x1_location_regression = (x_centers - temp_bbox_right_bound) / self.normalization_constant[i]
                        y1_location_regression = (y_centers - temp_bbox_bottom_bound) / self.normalization_constant[i]

                        temp_label[2, top_rf_center_index:bottom_rf_center_index + 1, \
                        left_rf_center_index : right_rf_center_index + 1] = \
                            np.tile(x0_location_regression, [bottom_rf_center_index - top_rf_center_index + 1, 1])

                        temp_label[3, top_rf_center_index:bottom_rf_center_index + 1,
                        left_rf_center_index:right_rf_center_index + 1] = \
                            np.tile(y0_location_regression, [right_rf_center_index - left_rf_center_index + 1, 1]).T

                        temp_label[4, top_rf_center_index:bottom_rf_center_index + 1,
                        left_rf_center_index:right_rf_center_index + 1] = \
                            np.tile(x1_location_regression, [bottom_rf_center_index - top_rf_center_index + 1, 1])

                        temp_label[5, top_rf_center_index:bottom_rf_center_index + 1,
                        left_rf_center_index:right_rf_center_index + 1] = \
                            np.tile(y1_location_regression, [right_rf_center_index - left_rf_center_index + 1, 1]).T

                score_gray_flag = np.logical_or(score_map_green > 1, score_map_gray > 0)
                location_green_flag = score_map_green == 1

                temp_label[0, :, :][location_green_flag] = 1
                temp_label[1, :, :][location_green_flag] = 0
                for c in range(self.num_output_channels):
                    if c == 0 or c == 1:
                        temp_mask[c, :, :][score_gray_flag] = 0
                        continue
                    # for bbox regression, only green area is available
                    temp_mask[c, :, :][location_green_flag] = 1

                label_list.append(temp_label)
                mask_list.append(temp_mask)

            im = im_input
            for n in range(self.num_output_scales):
                label_batch_list[n] = label_list[n]
                mask_batch_list[n] = mask_list[n]
            # if item == 0:
            #     print(im, " im")
            #     print(label_batch_list[0], " label0")
        else:
            """negative sample"""
            random_resize_factor = random.random() * \
                                   (self.neg_image_resize_factor_interval[1] - self.neg_image_resize_factor_interval[0]) \
                                   + self.neg_image_resize_factor_interval[0]

            im = cv2.resize(im, (0, 0), fy=random_resize_factor, fx=random_resize_factor)

            h_interval = im.shape[0] - self.net_input_height
            w_interval = im.shape[1] - self.net_input_width
            if h_interval >= 0:
                y_top = random.randint(0, h_interval)
            else:
                y_pad = int(-h_interval / 2)
            if w_interval >= 0:
                x_left = random.randint(0, w_interval)
            else:
                x_pad = int(-w_interval / 2)

            im_input = np.zeros((self.net_input_height, self.net_input_width, self.num_image_channels), dtype=np.uint8)

            if h_interval >= 0 and w_interval >= 0:
                im_input[:, :, :] = im[y_top:y_top + self.net_input_height, x_left:x_left + self.net_input_width, :]
            elif h_interval >= 0 and w_interval < 0:
                im_input[:, x_pad:x_pad + im.shape[1], :] = im[y_top:y_top + self.net_input_height, :, :]
            elif h_interval < 0 and w_interval >= 0:
                im_input[y_pad:y_pad + im.shape[0], :, :] = im[:, x_left:x_left + self.net_input_width, :]
            else:
                im_input[y_pad:y_pad + im.shape[0], x_pad:x_pad + im.shape[1], :] = im[:, :, :]

            # data augmentation
            if self.enable_horizon_flip and random.random() > 0.5:
                im_input = Augmentor.flip(im_input, 'h')
            if self.enable_vertical_flip and random.random() > 0.5:
                im_input = Augmentor.flip(im_input, 'v')

            if random.random() > 0.5:
                random.shuffle(self.pixel_augmentor_func_list)
                for augmentor in self.pixel_augmentor_func_list:
                    im_input = augmentor(im_input)

            im_input = im_input.astype(np.float32)
            im_input = im_input.transpose([2, 0, 1])
            im = im_input
            for label_batch in label_batch_list:
                label_batch[1, :, :] = 1
            for mask_batch in mask_batch_list:
                mask_batch[0:2, :, :] = 1

        return im, label_batch_list, mask_batch_list

    def __len__(self):
        return len(self.all_images)

    def _get_annotation(self, idx):
        NotImplementedError

    def _get_image(self, idx):
        NotImplementedError


class LFFDDatasetVOC(LFFDDatasetBase):

    def __init__(self,
                 rootfolder,
                 enable_horizon_flip,
                 enable_vertical_flip,
                 enable_random_brightness,
                 brightness_params,
                 enable_random_saturation,
                 saturation_params,
                 enable_random_contrast,
                 contrast_params,
                 enable_blur,
                 blur_params,
                 blur_kernel_size_list,
                 num_image_channels,
                 net_input_height,
                 net_input_width,
                 num_output_scales,
                 receptive_field_list,
                 receptive_field_stride,
                 feature_map_size_list,
                 receptive_field_center_start,
                 bbox_small_list,
                 bbox_large_list,
                 bbox_small_gray_list,
                 bbox_large_gray_list,
                 num_output_channels,
                 neg_image_resize_factor_interval
                 ):

        super(LFFDDatasetVOC, self).__init__(enable_horizon_flip,
                 enable_vertical_flip,
                 enable_random_brightness,
                 brightness_params,
                 enable_random_saturation,
                 saturation_params,
                 enable_random_contrast,
                 contrast_params,
                 enable_blur,
                 blur_params,
                 blur_kernel_size_list,
                 num_image_channels,
                 net_input_height,
                 net_input_width,
                 num_output_scales,
                 receptive_field_list,
                 receptive_field_stride,
                 feature_map_size_list,
                 receptive_field_center_start,
                 bbox_small_list,
                 bbox_large_list,
                 bbox_small_gray_list,
                 bbox_large_gray_list,
                 num_output_channels,
                 neg_image_resize_factor_interval)
        self.rootfolder = rootfolder
        self.face_paths = glob.glob(self.rootfolder + "/faces/*.jpg")
        self.noface_paths = glob.glob(self.rootfolder + "/nofaces/*.jpg")
        self.all_images = self.face_paths + self.noface_paths
        self.labels = [label.replace(".jpg", ".xml").replace("faces", "labels") for label in self.face_paths]
        self.class_names = ("__background__", "basketball", "volleyball")
        self.positive_index = [x for x in range(len(self.face_paths))]
        self.negative_index = [x + len(self.face_paths) for x in range(len(self.noface_paths))]

    def __len__(self):
        return len(self.all_images)

    def _get_annotation(self, idx):
        boxes = []
        labels = []
        if idx >= len(self.face_paths):
            return boxes, None
        annotation_file = self.labels[idx]
        objects = ET.parse(annotation_file).findall("object")
        size = ET.parse(annotation_file).find("size")
        #is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            imgW = float(size.find('width').text)
            imgH = float(size.find('height').text)
            bw = x2 - x1
            bh = y2 - y1
            boxes.append([x1, y1, bw, bh])
            # labels.append(self.class_names.index(class_name))
        return boxes, 1

    def _get_image(self, idx):
        assert (idx < len(self.all_images), "{} vs {}".format(idx, len(self.all_images)))
        im = cv2.imread(self.all_images[idx])
        return im


class LFFDDatasetPKL(LFFDDatasetBase):

    def __init__(self,
                 pickle_file_path,
                 enable_horizon_flip,
                 enable_vertical_flip,
                 enable_random_brightness,
                 brightness_params,
                 enable_random_saturation,
                 saturation_params,
                 enable_random_contrast,
                 contrast_params,
                 enable_blur,
                 blur_params,
                 blur_kernel_size_list,
                 num_image_channels,
                 net_input_height,
                 net_input_width,
                 num_output_scales,
                 receptive_field_list,
                 receptive_field_stride,
                 feature_map_size_list,
                 receptive_field_center_start,
                 bbox_small_list,
                 bbox_large_list,
                 bbox_small_gray_list,
                 bbox_large_gray_list,
                 num_output_channels,
                 neg_image_resize_factor_interval,
                 encode_quality=90
                 ):

        super(LFFDDatasetPKL, self).__init__(
                 enable_horizon_flip,
                 enable_vertical_flip,
                 enable_random_brightness,
                 brightness_params,
                 enable_random_saturation,
                 saturation_params,
                 enable_random_contrast,
                 contrast_params,
                 enable_blur,
                 blur_params,
                 blur_kernel_size_list,
                 num_image_channels,
                 net_input_height,
                 net_input_width,
                 num_output_scales,
                 receptive_field_list,
                 receptive_field_stride,
                 feature_map_size_list,
                 receptive_field_center_start,
                 bbox_small_list,
                 bbox_large_list,
                 bbox_small_gray_list,
                 bbox_large_gray_list,
                 num_output_channels,
                 neg_image_resize_factor_interval)
        self.data = pickle.load(open(pickle_file_path, 'rb'))
        # get positive and negative indeices
        self._positive_index = []
        self._negative_index = []
        for k, v in self.data.items():
            if v[1] == 0:  # negative
                self._negative_index.append(k)
            else: # positive
                self._positive_index.append(k)
        # self._positive_index.append(0)
        # self._negative_index.append(279)

        self.compression_mode = '.jpg'
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, encode_quality]

    def __len__(self):
        return len(self.positive_index) + len(self.negative_index)

    def _get_annotation(self, index):
        _, flag, bboxes = self.data[index]
        if flag == 0:
            return bboxes, None
        else:
            return bboxes, 1

    def _get_image(self, index):
        im_buf, _, _ = self.data[index]
        im = cv2.imdecode(im_buf, cv2.IMREAD_COLOR)
        cv2.imwrite("/opt/pytorch_{}.jpg".format(index), im)
        return im

    @property
    def positive_index(self):
        return self._positive_index

    @property
    def negative_index(self):
        return self._negative_index

class LFFDBatchSampler(Sampler):

    def __init__(self,
                 dataset,
                 batch_size,
                 num_neg_images_per_batch,
                 max_iter):
        super(Sampler, self).__init__()
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_neg_images_per_batch = num_neg_images_per_batch
        self.max_iter = max_iter
        assert self.num_neg_images_per_batch < self.batch_size

    def __iter__(self):
        batch = []
        for i in range(self.max_iter):
            loop = 0
            while loop < self.batch_size:

                if loop < self.num_neg_images_per_batch:
                    rand_idx = random.choice(self.dataset.negative_index)
                    batch.append(rand_idx)
                else:
                    rand_idx = random.choice(self.dataset.positive_index)
                    batch.append(rand_idx)
                loop +=1
            yield batch
            batch = []

    def __len__(self):
        return (self.max_iter + self.batch_size - 1) // self.batch_size


def lffd_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    imagbatch = torch.cat([torch.from_numpy(t[0]).unsqueeze(0) for t in batch], dim=0)
    branch1_label = torch.cat([torch.from_numpy(t[1][0]).unsqueeze(0) for t in batch], dim=0)
    branch2_label = torch.cat([torch.from_numpy(t[1][1]).unsqueeze(0) for t in batch], dim=0)
    branch3_label = torch.cat([torch.from_numpy(t[1][2]).unsqueeze(0) for t in batch], dim=0)
    branch4_label = torch.cat([torch.from_numpy(t[1][3]).unsqueeze(0) for t in batch], dim=0)
    branch5_label = torch.cat([torch.from_numpy(t[1][4]).unsqueeze(0) for t in batch], dim=0)
    branch6_label = torch.cat([torch.from_numpy(t[1][5]).unsqueeze(0) for t in batch], dim=0)
    branch7_label = torch.cat([torch.from_numpy(t[1][6]).unsqueeze(0) for t in batch], dim=0)
    branch8_label = torch.cat([torch.from_numpy(t[1][7]).unsqueeze(0) for t in batch], dim=0)

    branch1_mask = torch.cat([torch.from_numpy(t[2][0]).unsqueeze(0) for t in batch], dim=0)
    branch2_mask = torch.cat([torch.from_numpy(t[2][1]).unsqueeze(0) for t in batch], dim=0)
    branch3_mask = torch.cat([torch.from_numpy(t[2][2]).unsqueeze(0) for t in batch], dim=0)
    branch4_mask = torch.cat([torch.from_numpy(t[2][3]).unsqueeze(0) for t in batch], dim=0)
    branch5_mask = torch.cat([torch.from_numpy(t[2][4]).unsqueeze(0) for t in batch], dim=0)
    branch6_mask = torch.cat([torch.from_numpy(t[2][5]).unsqueeze(0) for t in batch], dim=0)
    branch7_mask = torch.cat([torch.from_numpy(t[2][6]).unsqueeze(0) for t in batch], dim=0)
    branch8_mask = torch.cat([torch.from_numpy(t[2][7]).unsqueeze(0) for t in batch], dim=0)
    return imagbatch, branch1_label, branch2_label, branch3_label, branch4_label, branch5_label, branch6_label, branch7_label, branch8_label, branch1_mask, branch2_mask, branch3_mask, branch4_mask, branch5_mask, branch6_mask, branch7_mask, branch8_mask



class BasketDataset(Dataset):
    def __init__(self,img_path,transform = None):
        self.img_paths = glob.glob(img_path + "/images/*.jpg")
        self.labels = [label.replace(".jpg",".xml").replace("images","labels") for label in self.img_paths]
        self.class_names = ("__background__","basketball","volleyball")
        prior = Priors() 
        self.center_form_priors = prior() # center form
        self.imgW,self.imgH = 512,512
        self.corner_form_priors = center_form_to_corner_form(self.center_form_priors)
        self.transform = transform  
    def __getitem__(self,idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label_file = self.labels[idx]
        gt_bboxes,gt_classes = self._get_annotation(idx)
        
        if self.transform:
            img,gt_bboxes,gt_classes = self.transform(np.array(img),gt_bboxes,gt_classes)
        #if self.transform:
        #    img = self.transform(img)
        gt_bboxes = torch.tensor(gt_bboxes)
        gt_classes = torch.LongTensor(gt_classes)
        
        gt_bboxes,gt_classes = assign_priors(gt_bboxes,gt_classes,self.center_form_priors,0.5) # corner form
        
        #imH,imW = cv_img.shape[:2]
        
        #gt_bboxes = corner_form_to_center_form(gt_bboxes) # (1524, 4) center form
        locations = convert_boxes_to_locations(gt_bboxes, self.center_form_priors,2) # 相当于归一化 corner_form
        # 拟合距离而不是直接拟合，这样更容易拟合。
        
        return [img,locations,gt_classes]
    def _get_annotation(self,idx):
        annotation_file = self.labels[idx]
        objects = ET.parse(annotation_file).findall("object")
        size = ET.parse(annotation_file).find("size")
        boxes = []
        labels = []
        #is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            imgW = float(size.find('width').text)
            imgH = float(size.find('height').text)
            boxes.append([x1/imgW,y1/imgH,x2/imgW,y2/imgH])
            labels.append(self.class_names.index(class_name))
        return boxes,labels
        #return (torch.tensor(boxes, dtype=torch.float),
        #        torch.tensor(labels, dtype=torch.long))
    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    # trigger for horizon flip
    param_enable_horizon_flip = True

    # trigger for vertical flip
    param_enable_vertical_flip = False

    # trigger for brightness
    param_enable_random_brightness = True
    param_brightness_factors = {'min_factor': 0.5, 'max_factor': 1.5}

    # trigger for saturation
    param_enable_random_saturation = True
    param_saturation_factors = {'min_factor': 0.5, 'max_factor': 1.5}

    # trigger for contrast
    param_enable_random_contrast = True
    param_contrast_factors = {'min_factor': 0.5, 'max_factor': 1.5}

    # trigger for blur
    param_enable_blur = False
    param_blur_factors = {'mode': 'random', 'sigma': 1}
    param_blur_kernel_size_list = [3]

    # negative image resize interval
    param_neg_image_resize_factor_interval = [0.5, 3.5]

    '''
        algorithm
    '''
    # the number of image channels
    param_num_image_channel = 3

    # the number of output scales (loss branches)
    param_num_output_scales = 8

    # feature map size for each scale
    param_feature_map_size_list = [159, 159, 79, 79, 39, 19, 19, 19]

    # bbox lower bound for each scale
    param_bbox_small_list = [10, 15, 20, 40, 70, 110, 250, 400]
    assert len(param_bbox_small_list) == param_num_output_scales

    # bbox upper bound for each scale
    param_bbox_large_list = [15, 20, 40, 70, 110, 250, 400, 560]
    assert len(param_bbox_large_list) == param_num_output_scales

    # bbox gray lower bound for each scale
    param_bbox_small_gray_list = [math.floor(v * 0.9) for v in param_bbox_small_list]
    # bbox gray upper bound for each scale
    param_bbox_large_gray_list = [math.ceil(v * 1.1) for v in param_bbox_large_list]

    # the RF size of each scale used for normalization, here we use param_bbox_large_list for better regression
    param_receptive_field_list = param_bbox_large_list
    # RF stride for each scale
    param_receptive_field_stride = [4, 4, 8, 8, 16, 32, 32, 32]
    # the start location of the first RF of each scale
    param_receptive_field_center_start = [3, 3, 7, 7, 15, 31, 31, 31]

    # the sum of the number of output channels, 2 channels for classification and 4 for bbox regression
    param_num_output_channels = 6

    # the ratio of neg image in a batch
    param_neg_image_ratio = 0.1

    # input height for network
    param_net_input_height = 640

    # input width for network
    param_net_input_width = 640

    # db = LFFDDatasetVOC(rootfolder="./datasets",
    #                     enable_horizon_flip=param_enable_horizon_flip,
    #                     enable_vertical_flip=param_enable_vertical_flip,
    #                     enable_random_brightness=param_enable_random_brightness,
    #                     brightness_params=param_brightness_factors,
    #                     enable_random_saturation=param_enable_random_saturation,
    #                     saturation_params=param_saturation_factors,
    #                     enable_random_contrast=param_enable_random_contrast,
    #                     contrast_params=param_contrast_factors,
    #                     enable_blur=param_enable_blur,
    #                     blur_params=param_blur_factors,
    #                     blur_kernel_size_list=param_blur_kernel_size_list,
    #                     num_image_channels=param_num_image_channel,
    #                     net_input_height=param_net_input_height,
    #                     net_input_width=param_net_input_width,
    #                     num_output_scales=param_num_output_scales,
    #                     receptive_field_list=param_receptive_field_list,
    #                     receptive_field_stride=param_receptive_field_stride,
    #                     feature_map_size_list=param_feature_map_size_list,
    #                     receptive_field_center_start=param_receptive_field_center_start,
    #                     bbox_small_list=param_bbox_small_list,
    #                     bbox_large_list=param_bbox_large_list,
    #                     bbox_small_gray_list=param_bbox_small_gray_list,
    #                     bbox_large_gray_list=param_bbox_large_gray_list,
    #                     num_output_channels=param_num_output_channels,
    #                     neg_image_resize_factor_interval=param_neg_image_resize_factor_interval)

    dbpkl = LFFDDatasetPKL(pickle_file_path="./datasets/widerface_train_data_gt_8.pkl",
                     enable_horizon_flip=param_enable_horizon_flip,
                     enable_vertical_flip=param_enable_vertical_flip,
                     enable_random_brightness=param_enable_random_brightness,
                     brightness_params=param_brightness_factors,
                     enable_random_saturation=param_enable_random_saturation,
                     saturation_params=param_saturation_factors,
                     enable_random_contrast=param_enable_random_contrast,
                     contrast_params=param_contrast_factors,
                     enable_blur=param_enable_blur,
                     blur_params=param_blur_factors,
                     blur_kernel_size_list=param_blur_kernel_size_list,
                     num_image_channels=param_num_image_channel,
                     net_input_height=param_net_input_height,
                     net_input_width=param_net_input_width,
                     num_output_scales=param_num_output_scales,
                     receptive_field_list=param_receptive_field_list,
                     receptive_field_stride=param_receptive_field_stride,
                     feature_map_size_list=param_feature_map_size_list,
                     receptive_field_center_start=param_receptive_field_center_start,
                     bbox_small_list=param_bbox_small_list,
                     bbox_large_list=param_bbox_large_list,
                     bbox_small_gray_list=param_bbox_small_gray_list,
                     bbox_large_gray_list=param_bbox_large_gray_list,
                     num_output_channels=param_num_output_channels,
                     neg_image_resize_factor_interval=param_neg_image_resize_factor_interval)

    batchsampler = LFFDBatchSampler(dataset=dbpkl,
                                    batch_size=4,
                                    num_neg_images_per_batch=1,
                                    max_iter=1000)

    loader = dataloader.DataLoader(dataset=dbpkl,
                                       batch_sampler=batchsampler, timeout=30,
                                   collate_fn=lffd_collate, num_workers=1)

    index = 0
    for imagbatch, branch1_label, branch2_label, branch3_label, branch4_label, branch5_label, branch6_label, branch7_label, branch8_label, branch1_mask, branch2_mask, branch3_mask, branch4_mask, branch5_mask, branch6_mask, branch7_mask, branch8_mask in loader:
        # pass
        print(imagbatch.shape)
        # print(index)
        # index +=1


