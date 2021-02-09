import os
import random
import json
import shutil
from memory_tempfile import MemoryTempfile
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

__all__ = ['COCODetection']

class COCODetection():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        assert 'path' in kwargs and 'split' in kwargs, 'kwargs must have path and split'

        dataset_folders = os.listdir(kwargs['path'])
        assert 'annotations' in dataset_folders, 'invalid path to coco dataset annotations'
        annotations_dir = os.path.join(kwargs['path'], 'annotations')

        shuffle = kwargs.get('shuffle', False)
        image_base_dir = 'images' if ('images' in dataset_folders) else ''
        image_base_dir = os.path.join(kwargs['path'], image_base_dir)
        image_split_dirs = os.listdir(image_base_dir)
        assert kwargs['split'] in image_split_dirs, f'invalid path to coco dataset images/split {kwargs["split"]}'
        self.image_dir = os.path.join(image_base_dir, kwargs['split'])

        self.coco_dataset = COCO(os.path.join(annotations_dir, f'instances_{kwargs["split"]}.json'))

        filter_imgs = kwargs['filter_imgs'] if 'filter_imgs' in kwargs else None
        if isinstance(filter_imgs, str):
            # filter images with the given list
            filter_imgs = os.path.join(kwargs['path'], filter_imgs)
            with open(filter_imgs) as filter_fp:
                filter = [int(id) for id in list(filter_fp)]
                orig_keys = list(self.coco_dataset.imgs)
                orig_keys = [k for k in orig_keys if k in filter]
                self.coco_dataset.imgs = {k: self.coco_dataset.imgs[k] for k in orig_keys}
            #
        elif filter_imgs:
            # filter and use images with gt only
            sel_keys = []
            for img_key, img_anns in self.coco_dataset.imgToAnns.items():
                if len(img_anns) > 0:
                    sel_keys.append(img_key)
                #
            #
            self.coco_dataset.imgs = {k: self.coco_dataset.imgs[k] for k in sel_keys}
        #

        max_frames = len(self.coco_dataset.imgs)
        num_frames = kwargs.get('num_frames', None)
        num_frames = min(num_frames, max_frames) if num_frames is not None else max_frames

        imgs_list = list(self.coco_dataset.imgs.items())
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(imgs_list)
        #
        self.coco_dataset.imgs = {k:v for k,v in imgs_list[:num_frames]}

        self.cat_ids = self.coco_dataset.getCatIds()
        self.img_ids = self.coco_dataset.getImgIds()
        self.num_frames = num_frames
        self.tempfiles = []

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = self.coco_dataset.loadImgs([img_id])[0]
        image_path = os.path.join(self.image_dir, img['file_name'])
        return image_path

    def __len__(self):
        return self.num_frames

    def __del__(self):
        for t in self.tempfiles:
            if os.path.exists(t):
                shutil.rmtree(t)

    def __call__(self, predictions, **kwargs):
        return self.evaluate(predictions, **kwargs)

    def evaluate(self, predictions, **kwargs):
        label_offset = kwargs.get('label_offset_pred', 0)
        work_dir = kwargs.get('work_dir', None)
        if work_dir is None:
            temp_dir_mem = MemoryTempfile()
            work_dir = temp_dir_mem.tempdir if hasattr(temp_dir_mem, 'tempdir') else temp_dir_mem.name
            self.tempfiles.append(work_dir)
        #
        os.makedirs(work_dir, exist_ok=True)
        detections_formatted_list = []
        for frame_idx, det_frame in enumerate(predictions):
            for det_id, det in enumerate(det_frame):
                det = self._format_detections(det, frame_idx, label_offset=label_offset)
                category_id = det['category_id'] if isinstance(det, dict) else det[4]
                if category_id >= 1: # final coco categories start from 1
                    detections_formatted_list.append(det)
                #
            #
        #
        coco_ap = 0.0
        if len(detections_formatted_list) > 0:
            detection_file = os.path.join(work_dir, 'detection_results.json')
            with open(detection_file, 'w') as det_fp:
                json.dump(detections_formatted_list, det_fp)
            #
            cocoDet = self.coco_dataset.loadRes(detection_file)
            cocoEval = COCOeval(self.coco_dataset, cocoDet, iouType='bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            coco_ap = cocoEval.stats[0]
        #
        accuracy = {'COCO Det AP[.5:.95]%': coco_ap}
        return accuracy

    def _format_detections(self, bbox_label_score, image_id, label_offset=0, class_map=None):
        if class_map is not None:
            assert bbox_label_score[4] in class_map, 'invalid prediction label or class_map'
            bbox_label_score[4] = class_map[bbox_label_score[4]]
        #
        bbox_label_score[4] = self._detection_label_to_catid(bbox_label_score[4], label_offset)
        output_dict = dict()
        image_id = self.img_ids[image_id]
        output_dict['image_id'] = image_id
        det_bbox = bbox_label_score[:4]      # json is not support for ndarray - convert to list
        det_bbox = self._xyxy2xywh(det_bbox) # can also be done in postprocess pipeline
        det_bbox = self._to_list(det_bbox)
        output_dict['bbox'] = det_bbox
        output_dict['category_id'] = int(bbox_label_score[4])
        output_dict['score'] = float(bbox_label_score[5])
        return output_dict

    def _detection_label_to_catid(self, label, label_offset):
        if isinstance(label_offset, (list,tuple)):
            label = int(label)
            assert label<len(label_offset), 'label_offset is a list/tuple, but its size is smaller than the detected label'
            label = label_offset[label]
        elif isinstance(label_offset, dict):
            label = int(label)
            assert label in label_offset.keys(), f'label_offset is a dict, but the detected label {label} was not one of its keys'
            label = label_offset[label]
        else:
            label = int(label - label_offset)
            assert label<len(self.cat_ids), \
                'the detected label could not be mapped to the 90 COCO categories using the default COCO.getCatIds()'
            label = self.cat_ids[label]
        #
        return label

    def _to_list(self, bbox):
        bbox = [float(x) for x in bbox]
        return bbox

    def _xyxy2xywh(self, bbox):
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        return bbox

