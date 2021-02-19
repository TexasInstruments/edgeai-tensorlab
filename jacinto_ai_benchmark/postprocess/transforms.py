import os
import copy
import numpy as np
import cv2
from PIL import ImageDraw


##############################################################################
class IndexArray():
    def __init__(self, index=0):
        self.index = index

    def __call__(self, input, info_dict):
        return input[self.index], info_dict


class ArgMax():
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, tensor, info_dict):
        if self.axis is None:
            axis = 1 if tensor.ndim == 4 else 0
        else:
            axis = self.axis
        #
        if tensor.shape[axis] > 1:
            tensor = tensor.argmax(axis=axis)
            tensor = tensor[0]
        #
        return tensor, info_dict


class Concat():
    def __init__(self, axis=-1, start_index=0, end_index=-1):
        self.axis = axis
        self.start_index = start_index
        self.end_index = end_index

    def __call__(self, tensor_list, info_dict):
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
            tensor = np.concatenate(tensor_list[self.start_index:self.end_index], axis=self.axis)
        else:
            tensor = tensor_list
        #
        return tensor, info_dict


##############################################################################
class SegmentationImageResize():
    def __call__(self, label, info_dict):
        image_shape = info_dict['data_shape']
        label = cv2.resize(label, dsize=(image_shape[1],image_shape[0]), interpolation=cv2.INTER_NEAREST)
        return label, info_dict


class SegmentationImageSave():
    def __init__(self):
        self.colors = [(r,g,b) for r in range(0,256,32) for g in range(0,256,32) for b in range(0,256,32)]

    def __call__(self, tensor, info_dict):
        data_path = info_dict['data_path']
        # img_data = info_dict['data']
        image_name = os.path.split(data_path)[-1]
        run_dir = info_dict['run_dir']
        save_dir = os.path.join(run_dir, 'segmentation')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, image_name)

        # TODO: convert label to color here
        if isinstance(tensor, np.ndarray):
            # convert image to BGR
            tensor = tensor[:,:,::-1] if tensor.ndim > 2 else tensor
            cv2.imwrite(save_path, tensor)
        else:
            # add fill code here
            tensor.save(save_path)
        #
        return tensor, info_dict


##############################################################################
class DetectionResize():
    def __call__(self, bbox, info_dict):
        data_shape = info_dict['data_shape']
        # avoid accidental overflow
        bbox = bbox.clip(-1e6, 1e6)
        # apply scaling
        bbox[...,0] *= data_shape[1]
        bbox[...,1] *= data_shape[0]
        bbox[...,2] *= data_shape[1]
        bbox[...,3] *= data_shape[0]
        return bbox, info_dict

class DetectionFilter():
    def __init__(self, detection_thr):
        self.detection_thr = detection_thr

    def __call__(self, bbox, info_dict):
        if self.detection_thr is not None:
            bbox_score = bbox[:,5]
            bbox_selected = bbox_score >= self.detection_thr
            bbox = bbox[bbox_selected,...]
        #
        return bbox, info_dict


class DetectionFormatting():
    def __init__(self, dst_indices=(0,1,2,3), src_indices=(1,0,3,2)):
        self.src_indices = src_indices
        self.dst_indices = dst_indices

    def __call__(self, bbox, info_dict):
        bbox_copy = copy.deepcopy(bbox)
        bbox_copy[...,self.dst_indices] = bbox[...,self.src_indices]
        return bbox_copy, info_dict


DetectionXYXY2YXYX = DetectionFormatting
DetectionYXYX2XYXY = DetectionFormatting
DetectionYXHW2XYWH = DetectionFormatting


class DetectionXYXY2XYWH():
    def __call__(self, bbox, info_dict):
        w = bbox[...,2] - bbox[...,0]
        h = bbox[...,3] - bbox[...,1]
        bbox[...,2] = w
        bbox[...,3] = h
        return bbox, info_dict


class DetectionXYWH2XYXY():
    def __call__(self, bbox, info_dict):
        x2 = bbox[...,0] + bbox[...,2]
        y2 = bbox[...,1] + bbox[...,3]
        bbox[...,2] = x2
        bbox[...,3] = y2
        return bbox, info_dict


class DetectionImageSave():
    def __init__(self):
        self.colors = [(r,g,b) for r in range(0,256,32) for g in range(0,256,32) for b in range(0,256,32)]
        self.thickness = 2

    def __call__(self, bbox, info_dict):
        data_path = info_dict['data_path']
        img_data = info_dict['data']
        image_name = os.path.split(data_path)[-1]
        run_dir = info_dict['run_dir']
        save_dir = os.path.join(run_dir, 'detection')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, image_name)

        img_data = copy.deepcopy(img_data)
        if isinstance(img_data, np.ndarray):
            for bbox_one in bbox:
                label = int(bbox_one[4])
                label_color = self.colors[label % len(self.colors)]
                pt1 = (int(bbox_one[0]),int(bbox_one[1]))
                pt2 = (int(bbox_one[2]),int(bbox_one[3]))
                cv2.rectangle(img_data, pt1, pt2, color=label_color, thickness=self.thickness)
            #
            cv2.imwrite(save_path, img_data[:,:,::-1])
        else:
            img_rect = ImageDraw.Draw(img_data)
            for bbox_one in bbox:
                label = int(bbox_one[4])
                label_color = self.colors[label % len(self.colors)]
                rect = (int(bbox_one[0]),int(bbox_one[1]),int(bbox_one[2]),int(bbox_one[3]))
                img_rect.rectangle(rect, outline=label_color, width=self.thickness)
            #
            img_data.save(save_path)
        #
        return bbox, info_dict



##############################################################################
class NPTensorToImage(object):
    def __init__(self, data_layout='NCHW'):
        self.data_layout = data_layout

    def __call__(self, tensor, info_dict):
        assert isinstance(tensor, np.ndarray), 'input tensor must be an array'
        if tensor.ndim >= 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        #
        if tensor.ndim==2:
            if self.data_layout=='NHWC':
                tensor = tensor[..., np.newaxis]
            else:
                tensor = tensor[np.newaxis, ...]
        assert tensor.ndim == 3, 'could not convert to image'
        tensor = np.transpose(tensor, (1,2,0)) if self.data_layout == 'NCHW' else tensor
        assert tensor.shape[2] in (1,3), 'invalid number of channels'
        return tensor, info_dict

    def __repr__(self):
        return self.__class__.__name__ + f'({self.data_layout})'