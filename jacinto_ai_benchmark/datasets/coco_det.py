import os
import random
import json
import tempfile
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
        image_dir = os.path.join(image_base_dir, kwargs['split'])

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

        num_frames = kwargs.get('num_frames', None)
        if num_frames is not None:
            orig_keys = list(self.coco_dataset.imgs)[:num_frames]
            self.coco_dataset.imgs = {k:self.coco_dataset.imgs[k] for k in orig_keys}
        #
        self.cat_ids = self.coco_dataset.getCatIds()
        self.img_ids = self.coco_dataset.getImgIds()
        imgs = []
        for img_id in self.img_ids:
            img = self.coco_dataset.loadImgs([img_id])[0]
            imgs.append(os.path.join(image_dir, img['file_name']))
        #
        self.imgs = imgs
        self.num_frames = min(num_frames, len(self.imgs)) if (num_frames is not None) else len(self.imgs)
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.imgs)
        #

    def __getitem__(self, idx):
        return self.imgs[idx]

    def __len__(self):
        return self.num_frames

    def __call__(self, predictions, label_offset=0, result_dir=None):
        return self.evaluate(predictions, label_offset, result_dir)

    def get_imgs(self):
        return self.imgs

    def evaluate(self, predictions, label_offset=0, result_dir=None):
        result_dir = tempfile.TemporaryDirectory().name if result_dir is None else result_dir
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
        if len(detections_formatted_list) > 0:
            detection_file = os.path.join(result_dir, 'detection_results.json')
            with open(detection_file, 'w') as det_fp:
                json.dump(detections_formatted_list, det_fp)
            #
            cocoDet = self.coco_dataset.loadRes(detection_file)
            cocoEval = COCOeval(self.coco_dataset, cocoDet, iouType='bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            coco_ap = cocoEval.stats[0]
        else:
            coco_ap = 0.0
        #
        return coco_ap

    def _format_detections(self, det_bbox_label_score, idx, format='json_serializable', label_offset=0, class_map=None):
        if class_map is not None:
            if det_bbox_label_score[4] in class_map.keys():
                det_bbox_label_score[4] = class_map[det_bbox_label_score[4]]
            else:
                print("Prediction of other classes")
            #
        #
        det_bbox_label_score[4] = self._detection_label_to_catid(det_bbox_label_score[4], label_offset)
        if format == 'json_serializable':
            output_dict = dict()
            image_id = self.img_ids[idx]
            output_dict['image_id'] = image_id
            output_dict['bbox'] = self._xyxy2xywh(det_bbox_label_score[:4])
            output_dict['category_id'] = int(det_bbox_label_score[4])
            output_dict['score'] = float(det_bbox_label_score[5])
            return output_dict
        else:
            return det_bbox_label_score

    def _detection_label_to_catid(self, label, label_offset):
        if isinstance(label_offset, (list,tuple)):
            label = int(label)
            assert label<len(label_offset), 'label_offset is a list/tuple, but its size is smaller than the detected label'
            label = label_offset[label]
        elif isinstance(label_offset, dict):
            label = int(label)
            assert label in label_offset.keys(), 'label_offset is a dict, but the detected label was not one of its keys'
            label = label_offset[label]
        else:
            label = int(label - label_offset)
            assert label<len(self.cat_ids), \
                'the detected label could not be mapped to the 90 COCO categories using the default COCO.getCatIds()'
            label = self.cat_ids[label]
        #
        return label

    def _xyxy2xywh(self, bbox):
        bbox = [float(x) for x in bbox]
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        return bbox

