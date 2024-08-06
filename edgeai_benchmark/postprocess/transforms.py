# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
import copy
import numbers

import PIL
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import numpy as np
import cv2
from munkres import Munkres
from numpy.lib.stride_tricks import as_strided
import math

from .. import constants
from .keypoints import *




##############################################################################
# utils
def get_font_cv2():
    font = cv2.FONT_HERSHEY_SIMPLEX
    return font


def get_font_pil():
    # font = PIL.ImageFont.truetype('arial.ttf', 10)
    font = PIL.ImageFont.load_default()
    return font


def apply_label_offset(label, label_offset):
    if label_offset is None:
        return label
    elif isinstance(label_offset, (list, tuple)):
        label = int(label)
        assert label < len(
            label_offset), 'label_offset is a list/tuple, but its size is smaller than the detected label'
        label = label_offset[label]
    elif isinstance(label_offset, dict):
        if np.isnan(label) or int(label) not in label_offset.keys():
            # print(utils.log_color('\nWARNING', 'detection incorrect', f'detected label: {label}'
            #                                                          f' is not in label_offset dict'))
            label = 0
        else:
            label = label_offset[int(label)]
        #
    elif isinstance(label_offset, numbers.Number):
        label = int(label + label_offset)
    else:
        label = int(label)
        assert label < len(self.cat_ids), \
            'the detected label could not be mapped to the 90 COCO categories using the default COCO.getCatIds()'
        label = self.cat_ids[label]
    #
    return label

def softmax(tensor,axis=-1):
    tensor = tensor - np.expand_dims(np.max(tensor, axis = axis), axis)
    tensor = np.exp(tensor)
    ax_sum = np.expand_dims(np.sum(tensor, axis = axis), axis)
    return tensor / ax_sum

##############################################################################
class SqueezeAxis():
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, input, info_dict):
        if isinstance(self.axis, (list,tuple)):
            return np.squeeze(input, self.axis)
        elif self.axis == 0:
            return input[self.axis], info_dict
        else:
            return np.squeeze(input, self.axis)


class ArgMax():
    def __init__(self, axis=None, data_layout=None):
        self.axis = axis
        self.data_layout = data_layout

    def __call__(self, tensor, info_dict):
        argmax_axis = None
        if self.axis is None:
            assert self.data_layout is not None, 'data_layout should not be None when axis in None'
            if self.data_layout == constants.NHWC:
                argmax_axis = -1
            elif self.data_layout == constants.NCHW:
                argmax_axis = ((tensor.ndim-3) if tensor.ndim >= 3  else None)
            #
        else:
            argmax_axis = self.axis
        #
        assert argmax_axis is not None, f'unsupport axis {self.axis} or data_layout {self.data_layout}'
        if tensor.shape[argmax_axis] > 1:
            tensor = tensor.argmax(axis=argmax_axis)
            tensor = tensor[0]
        #
        return tensor, info_dict


class Concat():
    def __init__(self, axis=-1, start_index=0, end_index=-1):
        self.axis = axis
        self.start_index = start_index
        self.end_index = end_index

    def __call__(self, tensor_list, info_dict):
        if isinstance(tensor_list, (list, tuple)):
            max_dim = 0
            for t_idx, t in enumerate(tensor_list):
                max_dim = max(max_dim, t.ndim)
            #
            for t_idx, t in enumerate(tensor_list):
                if t.ndim < max_dim:
                    tensor_list[t_idx] = t[..., np.newaxis]
                #
            #
            tensor = np.concatenate(tensor_list[self.start_index:self.end_index], axis=self.axis)
        else:
            tensor = tensor_list
        #
        return tensor, info_dict


class ShuffleList():
    def __init__(self, indices=None):
        self.indices = indices

    def __call__(self, tensor_list, info_dict):
        if self.indices is not None:
            tensor_list_out = []
            for ind in self.indices:
                tensor_list_out.append(tensor_list[ind])
            #
        else:
            tensor_list_out = tensor_list
        #
        return tensor_list_out, info_dict


class ReshapeList():
    def __init__(self, reshape_list=None):
        self.reshape_list = reshape_list

    def __call__(self, tensor_list, info_dict):
        keypoints = info_dict['dataset_info']['categories'][0].get('keypoints', None) if info_dict.get('dataset_info', None) else None
        if keypoints is not None:
            num_keypoints = len(info_dict['dataset_info']['categories'][0]['keypoints'])
            reshape_list = [(-1, 6+num_keypoints*3)]
            self.reshape_list = reshape_list
        if self.reshape_list is not None:
            tensor_list_out = []
            # if isinstance(self.reshape_list,tuple) and self.reshape_list[0] == 'detr' :
            #     tensor_list_softmax=[]
            #     tensor_list_softmax.append(tensor_list[1])
            #     tensor_list_argmax = np.argmax(tensor_list[0],axis=-1)
            #     softmax_score = softmax(tensor_list[0])[:,:,:-1]
            #     tensor_list_softmax.append(np.argmax(softmax_score,axis=-1))
            #     tensor_list_softmax.append(np.max(softmax_score,axis=-1))
            #     tensor_list = tensor_list_softmax
            #     for t_orig, t_shape in zip(tensor_list, self.reshape_list[1]):
            #         tensor_list_out.append(t_orig.reshape(t_shape))
            #     return tensor_list_out, info_dict
            for t_orig, t_shape in zip(tensor_list, self.reshape_list):
                tensor_list_out.append(t_orig.reshape(t_shape))
            #
        else:
            tensor_list_out = tensor_list
        #
        return tensor_list_out, info_dict
    

class IgnoreIndex():
    def __init__(self, indice=None):
        self.indice = indice

    def __call__(self, tensor, info_dict):
        if self.indice is not None:
            tensor_out = np.concatenate((tensor[..., :self.indice], tensor[..., self.indice + 1:]), -1)
        #
        else:
            tensor_out = tensor
        #
        return tensor_out, info_dict


class ClassificationImageSave():
    def __init__(self, num_output_frames=None):
        self.thickness = 2
        self.thickness_txt = 1
        self.dataset_info = None
        self.dataset_categories_map = None
        self.label_offset_pred = None
        self.num_output_frames = num_output_frames
        self.output_frame_idx = 0
        self.color_map = None

    def __call__(self, output, info_dict):
        if self.output_frame_idx >= self.num_output_frames:
            self.output_frame_idx += 1
            return output, info_dict
        #
        if self.color_map is None:
            self.color_map = info_dict['dataset_info']['color_map']
        #
        data_path = info_dict['data_path']
        img_data = info_dict['data']
        if self.label_offset_pred is None:
            self.label_offset_pred = info_dict.get('label_offset_pred', None)
        #
        if self.dataset_info is None:
            self.dataset_info = info_dict.get('dataset_info', None)
        #
        if self.dataset_info is not None:
            dataset_categories = self.dataset_info.get('categories', None)
            self.dataset_categories_map = {0: 'background'}
            self.dataset_categories_map.update({entry['id']: entry['name'] for entry in dataset_categories})
        #

        image_name = os.path.split(data_path)[-1]
        run_dir = info_dict['run_dir']
        save_dir = os.path.join(run_dir, 'outputs')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, image_name)
        img_data = copy.deepcopy(img_data)

        output_id = output[0] if isinstance(output, (list, tuple, np.ndarray)) else output
        output_id = output_id[0] if isinstance(output_id, (list, tuple, np.ndarray)) else output_id
        output_id = apply_label_offset(output_id, self.label_offset_pred)
        output_name = self.dataset_categories_map[output_id] if output_id in self.dataset_categories_map else output_id
        output_txt = f'category: {output_name}'
        label_color = self.color_map[output_id % len(self.color_map)]
        img_data = self.put_text(img_data, output_txt, label_color)
        if isinstance(img_data, np.ndarray):
            cv2.imwrite(save_path, img_data[:, :, ::-1])
        else:
            img_data.save(save_path)
        #
        self.output_frame_idx += 1
        return output, info_dict

    def put_text(self, img_data, output_txt, label_color):
        is_ndarray = isinstance(img_data, np.ndarray)
        img_data = np.array(img_data) if not is_ndarray else img_data
        pt = (20, 20)
        font_cv2 = get_font_cv2()
        font_scale = 0.5
        # fill background
        text_size = cv2.getTextSize(output_txt, font_cv2, fontScale=font_scale, thickness=self.thickness_txt)[0]
        cv2.rectangle(img_data, (pt[0], pt[1] - text_size[1]), (pt[0] + text_size[0], pt[1] + text_size[1] // 2),
                      (255, 255, 255), -1)
        # now write the actual text
        cv2.putText(img_data, output_txt, pt, font_cv2, fontScale=font_scale, color=label_color,
                    thickness=self.thickness_txt)
        img_data = PIL.Image.fromarray(img_data) if not is_ndarray else img_data
        return img_data


##############################################################################
class SegmentationImageResize():
    def __call__(self, label, info_dict):
        image_shape = info_dict['data_shape']
        if label.dtype in (np.int32, np.int64):
            label = label.astype(np.float32)
        #
        label = cv2.resize(label, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        return label, info_dict


class SegmentationImagetoBytes():
    '''
    Convert Segmentation image to bytes (uint8) to save space
    '''

    def __call__(self, label, info_dict):
        label = label.astype(np.uint8)
        return label, info_dict


class SegmentationImageSave():
    def __init__(self, num_output_frames=None, num_classes=None):
        self.num_classes = num_classes
        self.num_output_frames = num_output_frames
        self.output_frame_idx = 0
        self.color_map = None
        self.palette = None

    def update_color_map(self, color_map):
        self.color_map = color_map
        # convert label to color here
        self.palette = copy.deepcopy(color_map)
        for i, p in enumerate(self.palette):
            self.palette[i] = np.array(p, dtype=np.uint8)
            self.palette[i] = self.palette[i][..., ::-1]  # RGB->BGR, since palette is expected to be given in RGB format
        #
        self.palette = np.array(self.palette)

    def __call__(self, tensor, info_dict):
        if self.output_frame_idx >= self.num_output_frames:
            self.output_frame_idx += 1
            return tensor, info_dict
        #
        if self.color_map is None or self.palette is None:
            self.update_color_map(info_dict['dataset_info']['color_map'])
        #
        data_path = info_dict['data_path']
        # img_data = info_dict['data']
        image_name = os.path.split(data_path)[-1].split('.')[0] + '.png'
        run_dir = info_dict['run_dir']
        save_dir = os.path.join(run_dir, 'outputs')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, image_name)

        prediction = np.array(tensor, dtype=np.uint8)

        if len(prediction.shape) > 2 and prediction.shape[0] > 1:
            prediction = np.argmax(prediction, axis=0)

        prediction = np.squeeze(prediction)
        prediction_size = info_dict['data_shape']
        prediction = np.remainder(prediction, len(self.color_map))
        output_image = self.palette[prediction.ravel()].reshape(prediction_size)

        input_bgr = cv2.imread(data_path)  # Read the actual RGB image
        # if args.img_border_crop is not None:
        #    t, l, h, w = args.img_border_crop
        #    input_bgr = input_bgr[t:t + h, l:l + w]
        input_bgr = cv2.resize(input_bgr, dsize=(prediction.shape[1], prediction.shape[0]))
        output_image = self.chroma_blend(input_bgr, output_image)

        cv2.imwrite(save_path, output_image)
        if isinstance(output_image, np.ndarray):
            # convert image to BGR
            output_image = output_image[:, :, ::-1] if output_image.ndim > 2 else output_image
            cv2.imwrite(save_path, output_image)
        else:
            # add fill code here
            output_image.save(save_path)
        #
        self.output_frame_idx += 1
        return tensor, info_dict

    def chroma_blend(self, image, color, to_image_size=False):
        if image is None:
            return color
        elif color is None:
            return image
        #
        image_dtype = image.dtype
        color_dtype = color.dtype
        if image_dtype in (np.float32, np.float64):
            image = (image * 255).clip(0, 255).astype(np.uint8)
        #
        if color_dtype in (np.float32, np.float64):
            color = (color * 255).clip(0, 255).astype(np.uint8)
        #
        if image.shape != color.shape:
            if to_image_size:
                color = cv2.resize(color, dsize=(image.shape[1], image.shape[0]))
            else:
                image = cv2.resize(image, dsize=(color.shape[1], color.shape[0]))
            #
        #
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image_y, image_u, image_v = cv2.split(image_yuv)
        color_yuv = cv2.cvtColor(color, cv2.COLOR_BGR2YUV)
        color_y, color_u, color_v = cv2.split(color_yuv)
        image_y = np.uint8(image_y)
        color_u = np.uint8(color_u)
        color_v = np.uint8(color_v)
        image_yuv = cv2.merge((image_y, color_u, color_v))
        image = cv2.cvtColor(image_yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
        if image_dtype in (np.float32, np.float64):
            image = image / 255.0
        #
        return image


##############################################################################
class DetectionResizeOnlyNormalized():
    def __call__(self, bbox, info_dict):
        img_data = info_dict['data']
        assert isinstance(img_data, np.ndarray), 'only supports np array for now'
        data_shape = info_dict['data_shape']
        data_height, data_width, _ = data_shape
        # avoid accidental overflow
        bbox = bbox.clip(-1e6, 1e6)
        # scale the detections from normalized shape (0-1) to data shape
        bbox[..., 0] = (bbox[..., 0] * data_width).clip(0, data_width)
        bbox[..., 1] = (bbox[..., 1] * data_height).clip(0, data_height)
        bbox[..., 2] = (bbox[..., 2] * data_width).clip(0, data_width)
        bbox[..., 3] = (bbox[..., 3] * data_height).clip(0, data_height)
        return bbox, info_dict


class DetectionResizePad():
    def __init__(self, resize_with_pad=False, normalized_detections=True, keypoint=False, object6dpose=False):
        self.resize_with_pad = resize_with_pad
        self.normalized_detections = normalized_detections
        self.keypoint = keypoint
        self.object6dpose = object6dpose

    def __call__(self, bbox, info_dict):
        if 'data' in info_dict:
            img_data = info_dict['data']
            assert isinstance(img_data, np.ndarray), 'only supports np array for now'
        # avoid accidental overflow
        bbox = bbox.clip(-1e6, 1e6)
        # img size without pad
        data_shape = info_dict['data_shape']
        data_height, data_width, _ = data_shape
        if 'resize_shape' in info_dict:
            resize_shape = info_dict['resize_shape']
            resize_height, resize_width, _ = resize_shape
        if self.resize_with_pad:
            # account for padding
            border = info_dict['resize_border']
            left, top, right, bottom = border
            bbox[..., 0] -= left
            bbox[..., 1] -= top
            bbox[..., 2] -= left
            bbox[..., 3] -= top
            resize_height, resize_width = (resize_height - top - bottom), (resize_width - left - right)
            if self.keypoint:
                bbox[..., 6::3] -= left
                bbox[..., 7::3] -= top
        #
        # scale the detections from the input shape to data shape
        sh = data_height / (1.0 if self.normalized_detections else resize_height)
        sw = data_width / (1.0 if self.normalized_detections else resize_width)
        bbox[..., 0] = (bbox[..., 0] * sw).clip(0, data_width)
        bbox[..., 1] = (bbox[..., 1] * sh).clip(0, data_height)
        bbox[..., 2] = (bbox[..., 2] * sw).clip(0, data_width)
        bbox[..., 3] = (bbox[..., 3] * sh).clip(0, data_height)
        if self.keypoint:
            bbox[..., 6::3] = (bbox[..., 6::3] * sw).clip(0, data_width)
            bbox[..., 7::3] = (bbox[..., 7::3] * sh).clip(0, data_height)
        return bbox, info_dict


class DetectionFilter():
    def __init__(self, detection_threshold, detection_keep_top_k=None):
        self.detection_threshold = detection_threshold
        self.detection_keep_top_k = detection_keep_top_k

    def __call__(self, bbox, info_dict):
        if self.detection_threshold is not None:
            bbox_score = bbox[:, 5]
            bbox_selected = (bbox_score >= self.detection_threshold)
            bbox = bbox[bbox_selected, ...]
        #
        if self.detection_keep_top_k is not None and bbox.shape[0] > self.detection_keep_top_k:
            bbox = sorted(bbox, key=lambda b: b[5])
            bbox = np.stack(bbox, axis=0)
            bbox = bbox[range(self.detection_keep_top_k), ...]
        #
        return bbox, info_dict


class LogitsToLabelScore():
    def __init__(self, scores_index=0, bbox_index=1, background_class_id=-1):
        self.scores_index = scores_index
        self.bbox_index = bbox_index
        self.background_class_id = background_class_id

    def __call__(self, tensor_list, info_dict):
        tensor_list_softmax=[]
        if self.bbox_index is not None:
            tensor_list_softmax.append(tensor_list[self.bbox_index].reshape(-1,4))
        #
        softmax_score = softmax(tensor_list[self.scores_index])
        if self.background_class_id == -1:  
            softmax_score = softmax_score[...,:self.background_class_id]
        elif self.background_class_id is not None:
            softmax_score = softmax_score[...,self.background_class_id+1:]
        #
        tensor_list_softmax.append(np.argmax(softmax_score,axis=-1).reshape(-1,1))
        tensor_list_softmax.append(np.max(softmax_score,axis=-1).reshape(-1,1))
        return tensor_list_softmax, info_dict  
    

class DetectionFormatting():
    def __init__(self, dst_indices, src_indices):
        self.src_indices = src_indices
        self.dst_indices = dst_indices

    def __call__(self, bbox, info_dict):
        bbox_copy = copy.deepcopy(bbox)
        bbox_copy[..., self.dst_indices] = bbox[..., self.src_indices]
        return bbox_copy, info_dict


class DetectionXYXY2YXYX(DetectionFormatting):
    def __init__(self, dst_indices=(0, 1, 2, 3), src_indices=(1, 0, 3, 2)):
        super().__init__(dst_indices, src_indices)


class DetectionYXYX2XYXY(DetectionFormatting):
    def __init__(self, dst_indices=(0, 1, 2, 3), src_indices=(1, 0, 3, 2)):
        super().__init__(dst_indices, src_indices)


class DetectionYXHW2XYWH(DetectionFormatting):
    def __init__(self, dst_indices=(0, 1, 2, 3), src_indices=(1, 0, 3, 2)):
        super().__init__(dst_indices, src_indices)


class DetectionXYXY2XYWH():
    def __call__(self, bbox, info_dict):
        w = bbox[..., 2] - bbox[..., 0]
        h = bbox[..., 3] - bbox[..., 1]
        bbox[..., 2] = w
        bbox[..., 3] = h
        return bbox, info_dict


class DetectionXYWH2XYXY():
    def __call__(self, bbox, info_dict):
        x2 = bbox[..., 0] + bbox[..., 2]
        y2 = bbox[..., 1] + bbox[..., 3]
        bbox[..., 2] = x2
        bbox[..., 3] = y2
        return bbox, info_dict
    
class DetectionXYWH2XYXYCenterXY():
    def __call__(self, bbox, info_dict):
        x1 = bbox[..., 0] - 0.5 * bbox[..., 2]
        y1 = bbox[..., 1] - 0.5 * bbox[..., 3]
        x2 = bbox[..., 0] + 0.5 * bbox[..., 2]
        y2 = bbox[..., 1] + 0.5 * bbox[..., 3]
        img_shape =  info_dict['data_shape']
        resize_shape =  info_dict['resize_shape']
        bbox[..., 0] = x1 * resize_shape[1]
        bbox[..., 1] = y1 * resize_shape[0]
        bbox[..., 2] = x2 * resize_shape[1]
        bbox[..., 3] = y2 * resize_shape[0]
        return bbox, info_dict
    

class DetectionBoxSL2BoxLS(DetectionFormatting):
    def __init__(self, dst_indices=(4, 5), src_indices=(5, 4)):
        super().__init__(dst_indices, src_indices)

class Yolov4DetectionBoxSL2BoxLS(DetectionFormatting):
    def __init__(self, dst_indices=(0,1,2,3,4), src_indices=(1,2,3,4,0)):
        super().__init__(dst_indices, src_indices)


class DetectionImageSave():
    def __init__(self, num_output_frames=None):
        self.thickness = 2
        self.thickness_txt = 1
        self.dataset_info = None
        self.dataset_categories_map = None
        self.label_offset_pred = None
        self.num_output_frames = num_output_frames
        self.output_frame_idx = 0
        self.color_map = None

    def __call__(self, bbox, info_dict):
        if self.output_frame_idx >= self.num_output_frames:
            self.output_frame_idx += 1
            return bbox, info_dict
        #
        if self.color_map is None:
            self.color_map = info_dict['dataset_info']['color_map']
        #
        data_path = info_dict['data_path']
        img_data = info_dict['data']
        if self.label_offset_pred is None:
            self.label_offset_pred = info_dict.get('label_offset_pred', None)
        #
        if self.dataset_info is None:
            self.dataset_info = info_dict.get('dataset_info', None)
        #
        if self.dataset_info is not None:
            dataset_categories = self.dataset_info.get('categories', None)
            self.dataset_categories_map = {0: 'background'}
            self.dataset_categories_map.update({entry['id']: entry['name'] for entry in dataset_categories})
        #

        image_name = os.path.split(data_path)[-1]
        run_dir = info_dict['run_dir']
        save_dir = os.path.join(run_dir, 'outputs')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, image_name)
        img_data = copy.deepcopy(img_data)
        is_ndarray = isinstance(img_data, np.ndarray)
        img_data = np.array(img_data) if not is_ndarray else img_data
        for bbox_one in bbox:
            label = int(bbox_one[4])
            label_color = self.color_map[label % len(self.color_map)]
            pt1 = (int(bbox_one[0]), int(bbox_one[1]))
            pt2 = (int(bbox_one[2]), int(bbox_one[3]))
            label = apply_label_offset(label, self.label_offset_pred)
            output_name = self.dataset_categories_map[label] if label in self.dataset_categories_map else label
            output_txt = output_name
            img_data = self.put_text(img_data, (pt1[0], pt1[1] - 5), output_txt, label_color)
            img_data = self.put_rectangle(img_data, pt1, pt2, label_color)
        #
        img_data = PIL.Image.fromarray(img_data) if not is_ndarray else img_data
        if isinstance(img_data, np.ndarray):
            cv2.imwrite(save_path, img_data[:, :, ::-1])
        else:
            img_data.save(save_path)
        #
        self.output_frame_idx += 1
        return bbox, info_dict

    def put_text(self, img_data, pt, output_txt, label_color):
        is_ndarray = isinstance(img_data, np.ndarray)
        img_data = np.array(img_data) if not is_ndarray else img_data
        font_cv2 = get_font_cv2()
        font_scale = 0.5
        # fill background
        text_size = cv2.getTextSize(output_txt, font_cv2, fontScale=font_scale, thickness=self.thickness_txt)[0]
        cv2.rectangle(img_data, (pt[0], pt[1] - text_size[1]), (pt[0] + text_size[0], pt[1] + text_size[1] // 2),
                      (255, 255, 255), -1)
        # now write the actual text
        cv2.putText(img_data, output_txt, pt, font_cv2, fontScale=font_scale, color=label_color,
                    thickness=self.thickness_txt)
        img_data = PIL.Image.fromarray(img_data) if not is_ndarray else img_data
        return img_data

    def put_rectangle(self, img_data, pt1, pt2, label_color):
        font_cv2 = get_font_cv2()
        is_ndarray = isinstance(img_data, np.ndarray)
        img_data = np.array(img_data) if not is_ndarray else img_data
        cv2.rectangle(img_data, pt1, pt2, label_color, self.thickness)
        img_data = PIL.Image.fromarray(img_data) if not is_ndarray else img_data
        return img_data


##############################################################################
class NPTensorToImage(object):
    def __init__(self, data_layout='NCHW'):
        self.data_layout = data_layout

    def __call__(self, tensor, info_dict):
        assert isinstance(tensor, np.ndarray), 'input tensor must be an array'
        max_num_squeeze = 3
        for squeeze_index in range(max_num_squeeze):
            if tensor.ndim >= 3 and tensor.shape[0] == 1:
                tensor = tensor[0]
            #
        #
        if tensor.ndim == 2:
            if self.data_layout == 'NHWC':
                tensor = tensor[..., np.newaxis]
            else:
                tensor = tensor[np.newaxis, ...]
            #
        #
        assert tensor.ndim == 3, 'could not convert to image'
        tensor = np.transpose(tensor, (1, 2, 0)) if self.data_layout == 'NCHW' else tensor
        assert tensor.shape[2] in (1, 3), f'invalid number of channels. expected 1 or 3 channels, got {tensor.shape[2]}'
        return tensor, info_dict

    def __repr__(self):
        return self.__class__.__name__ + f'({self.data_layout})'


##############################################################################
class DepthImageResize():
    def __call__(self, label, info_dict):
        image_shape = info_dict['data_shape']
        # if label.dtype in (np.int32, np.int64):
        #     label = label.astype(np.float32)
        # #
        label = cv2.resize(label, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        return label, info_dict


class DepthImageSave():
    def __init__(self, num_output_frames=None):
        self.num_output_frames = num_output_frames
        self.output_frame_idx = 0

    # Taken from MiDaS (https://github.com/isl-org/MiDaS)
    def write_pfm(path, image, scale=1):
        """Write pfm file.

        Args:
            path (str): pathto file
            image (array): data
            scale (int, optional): Scale. Defaults to 1.
        """

        with open(path, "wb") as file:
            color = None

            if image.dtype.name != "float32":
                raise Exception("Image dtype must be float32.")

            image = np.flipud(image)

            if len(image.shape) == 3 and image.shape[2] == 3:  # color image
                color = True
            elif (
                    len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
            ):  # greyscale
                color = False
            else:
                raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

            file.write("PF\n" if color else "Pf\n".encode())
            file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

            endian = image.dtype.byteorder

            if endian == "<" or endian == "=" and sys.byteorder == "little":
                scale = -scale

            file.write("%f\n".encode() % scale)

            image.tofile(file)

    def _call_(self, result, info_dict):
        if self.output_frame_idx >= self.num_output_frames:
            self.output_frame_idx += 1
            return result, info_dict
        #
        data_path = info_dict['data_path']
        image_name = os.path.split(data_path)[-1]
        run_dir = info_dict['run_dir']
        save_dir = os.path.join(run_dir, 'outputs')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, image_name)

        pred = result['preds'].astype(np.float32)
        self.write_pfm(pred)

        # Write a relative 16 bit depth map
        d_min = np.min(pred)
        d_max = np.max(pred)
        pred_relative = 65535 * ((pred - d_min) / (d_max - d_min))

        cv2.imwrite(save_path, pred_relative.astype("uint16"))
        self.output_frame_idx += 1
        return result, info_dict


class OD3DOutPutPorcess(object):
    def __init__(self, detection_threshold):
        self.detection_threshold = detection_threshold

    def __call__(self, tidl_op, info_dict):
        selected_op = tidl_op[0][0][0]
        selected_op = selected_op[selected_op[:, 1] > self.detection_threshold]
        return np.array(selected_op), info_dict
