import os
import copy
import numpy as np
import cv2
from PIL import ImageDraw

class IndexArray():
    def __init__(self, index=0):
        self.index = index

    def __call__(self, input):
        return input[self.index]


class ArgMax():
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, tensor):
        if self.axis is None:
            axis = 1 if tensor.ndim == 4 else 0
        else:
            axis = self.axis
        #
        output = tensor.argmax(axis=axis)
        output = output[0]
        return output

class Concat():
    def __init__(self, axis=-1, start_inxdex=0, end_index=-1):
        self.axis = axis
        self.start_inxdex = start_inxdex
        self.end_index =end_index

    def __call__(self, tensor_list):
        if isinstance(tensor_list, (list,tuple)):
            max_dim = 0
            for t_idx, t in enumerate(tensor_list):
                max_dim = max(max_dim, t.ndim)
            #
            for t_idx, t in enumerate(tensor_list):
                if t.ndim < max_dim:
                    tensor_list[t_idx] = t[...,np.newaxis]
                #
            #
            tensor = np.concatenate(tensor_list[self.start_inxdex:self.end_index], axis=self.axis)
        else:
            tensor = tensor_list
        #
        return tensor


class DetectionResize():
    def __init__(self):
        self.image_shape = None

    def __call__(self, bbox):
        bbox[...,0] *= self.image_shape[1]
        bbox[...,1] *= self.image_shape[0]
        bbox[...,2] *= self.image_shape[1]
        bbox[...,3] *= self.image_shape[0]
        return bbox

    def set_info(self, info_dict):
        self.image_shape = info_dict['preprocess']['image_shape']


class DetectionFilter():
    def __init__(self, score_thr):
        self.score_thr = score_thr

    def __call__(self, bbox):
        if self.score_thr is not None:
            bbox_score = bbox[:,5]
            bbox_selected = bbox_score >= self.score_thr
            bbox = bbox[bbox_selected,...]
        #
        return bbox


class DetectionFormatting():
    def __init__(self, dst_indices=(0,1,2,3), src_indices=(1,0,3,2)):
        self.src_indices = src_indices
        self.dst_indices = dst_indices

    def __call__(self, bbox):
        bbox_copy = copy.deepcopy(bbox)
        bbox_copy[...,self.dst_indices] = bbox[...,self.src_indices]
        return bbox_copy


DetectionXYXY2YXYX = DetectionFormatting
DetectionYXYX2XYXY = DetectionFormatting
DetectionYXHW2XYWH = DetectionFormatting


class DetectionXYXY2XYWH():
    def __call__(self, bbox):
        w = bbox[...,2] - bbox[...,0]
        h = bbox[...,3] - bbox[...,1]
        bbox[...,2] = w
        bbox[...,3] = h
        return bbox


class DetectionXYWH2XYXY():
    def __call__(self, bbox):
        x2 = bbox[...,0] + bbox[...,2]
        y2 = bbox[...,1] + bbox[...,3]
        bbox[...,2] = x2
        bbox[...,3] = y2
        return bbox


class DetectionImageSave():
    def __init__(self):
        self.save_path = None
        self.color = (0,255,0)
        self.outline = 'green'
        self.thickness = 2

    def __call__(self, bbox):
        img = copy.deepcopy(self.image)
        if isinstance(img, np.ndarray):
            img = img[:,:,::-1] #to BGR
            for bbox_one in bbox:
                self.draw_bbox_cv2(img, bbox_one)
            #
        else:
            img_rect = ImageDraw.Draw(img)
            for bbox_one in bbox:
                self.draw_bbox_pil(img_rect, bbox_one)
            #
        #
        if isinstance(self.image, np.ndarray):
            cv2.imwrite(self.save_path, img)
        else:
            img.save(self.save_path)
        #
        return bbox

    def draw_bbox_cv2(self, img, bbox_one):
        cv2.rectangle(img, bbox_one[:2], bbox_one[2:4], color=self.color, thickness=self.thickness)
        return img

    def draw_bbox_pil(self, img, bbox_one):
        img.rectangle(bbox_one[:4], outline=self.outline, width=self.thickness)
        return img

    def set_info(self, info_dict):
        image_path = info_dict['preprocess']['image_path']
        self.image = info_dict['preprocess']['image']
        image_name = os.path.split(image_path)[-1]
        work_dir = info_dict['session']['work_dir']
        save_dir = os.path.join(work_dir, 'detections')
        os.makedirs(save_dir, exist_ok=True)
        self.save_path = os.path.join(save_dir, image_name)
