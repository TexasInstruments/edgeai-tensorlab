import json
import os
import random
import numpy as np
import copy
import cv2
from PIL import Image

from edgeai_torchmodelopt import xnn
from edgeai_tensorvision.xvision.datasets.dataset_utils import dataset_split

__all__ = ['common_segmentation', 'common_segmentation_with_background_class']


class CommonSegmentation():
    def __init__(self, root, split, shuffle=False, num_imgs=None, num_classes=None, with_background_class=False, **kwargs):
        from pycocotools.coco import COCO
        self.with_background_class = with_background_class
        self.class_names = None
        self.annotation_prefix = kwargs['annotation_prefix']

        dataset_folders = os.listdir(root)
        assert 'annotations' in dataset_folders, 'Invalid path to TI scape dataset annotations'
        annotations_dir = os.path.join(root, 'annotations')

        image_base_dir = 'images' if ('images' in dataset_folders) else ''
        image_base_dir = os.path.join(root, image_base_dir)
        image_dir = os.path.join(image_base_dir, '')

        self.tiscape_dataset = COCO(os.path.join(annotations_dir, f'{self.annotation_prefix}_{split}.json'))

        self.cat_ids = self.tiscape_dataset.getCatIds()
        if num_classes:
            assert num_classes == len(self.cat_ids), f'the provided num_classes={num_classes} does not match the length of cat_ids={self.cat_ids}'
        #
        self.num_classes_anno = len(self.cat_ids)
        self.min_class_id = min(self.cat_ids)
        if with_background_class and self.min_class_id > 0:
            self.num_classes_used = (self.num_classes_anno + 1)
        else:
            self.num_classes_used = self.num_classes_anno
        #
        self.categories = self.cat_ids

        img_ids = self.tiscape_dataset.getImgIds()
        self.img_ids = self._remove_images_without_annotations(img_ids)

        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.img_ids)

        if num_imgs is not None:
            self.img_ids = self.img_ids[:num_imgs]
            self.tiscape_dataset.imgs = {k: self.tiscape_dataset.imgs[k] for k in self.img_ids}

        imgs = []
        for img_id in self.img_ids:
            img = self.tiscape_dataset.loadImgs([img_id])[0]
            imgs.append(os.path.join(image_dir, img['file_name']))
        #
        self.imgs = imgs
        self.num_imgs = len(self.imgs)

    def __getitem__(self, idx, with_label=True):
        if with_label:
            image = Image.open(self.imgs[idx])
            ann_ids = self.tiscape_dataset.getAnnIds(imgIds=self.img_ids[idx], iscrowd=None)
            anno = self.tiscape_dataset.loadAnns(ann_ids)
            image, anno = self._filter_and_remap_categories(image, anno)
            image, target = self._convert_polys_to_mask(image, anno)
            image = np.array(image)
            if image.ndim == 2 or image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            #
            target = np.array(target)
            return image, target
        else:
            return self.imgs[idx]
        #

    def __len__(self):
        return self.num_imgs

    def _remove_images_without_annotations(self, img_ids):
        ids = []
        for ds_idx, img_id in enumerate(img_ids):
            ann_ids = self.tiscape_dataset.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.tiscape_dataset.loadAnns(ann_ids)
            if self.categories:
                anno = [obj for obj in anno if obj["category_id"] in self.categories]
            if self._has_valid_annotation(anno):
                ids.append(img_id)
            #
        #
        return ids

    def _has_valid_annotation(self, anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if more than 1k pixels occupied in the image
        return sum(obj["area"] for obj in anno) > 1000

    def _filter_and_remap_categories(self, image, anno, remap=True):
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not remap:
            return image, anno
        #
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])  + 1
    #
        return image, anno

    def _convert_polys_to_mask(self, image, anno):
        w, h = image.size
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if segmentations:
            masks = self._convert_poly_to_mask(segmentations, h, w)
            cats = np.array(cats, dtype=masks.dtype)
            cats = cats.reshape(-1, 1, 1)
            # merge all instance masks into a single segmentation map
            # with its corresponding categories
            target = (masks * cats).max(axis=0)
            # discard overlapping instances
            # target[masks.sum(0) > 1] = 255
        else:
            target = np.zeros((h, w), dtype=np.uint8)
        #
        return image, target

    def _convert_poly_to_mask(self, segmentations, height, width):
        from pycocotools import mask as coco_mask
        masks = []
        for polygons in segmentations:
            if len(polygons) != 1:
                polygons = [polygons]
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = mask.any(axis=2)
            mask = mask.astype(np.uint8)
            masks.append(mask)
        if masks:
            masks = np.stack(masks, axis=0)
        else:
            masks = np.zeros((0, height, width), dtype=np.uint8)
        return masks


class CommonSegmentationPlus(CommonSegmentation):
    def __init__(self, *args, num_classes=None, transforms=None, with_background_class=False, **kwargs):
        # 21 class is a special case, otherwise use all the classes
        # in get_item a modulo is done to map the target to the required num_classes
        super().__init__(*args, num_classes=num_classes, with_background_class=with_background_class, **kwargs)
        self.void_classes = []
        self.valid_classes = self.categories
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, self.valid_classes))
        self.color_map = xnn.utils.get_color_palette(self.num_classes_used)
        self.color_map = (self.color_map * self.num_classes_used)[:self.num_classes_used]
        self.label_colours = dict(zip(range(self.num_classes_used), self.color_map))
        self.transforms = transforms

    def __getitem__(self, item):
        image, target = super().__getitem__(item)
        image = [image]
        target = [target]
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        #
        if not self.with_background_class and self.min_class_id>0:
            # target[target==0] = self.num_classes_used
            # target = np.remainder(target, self.num_classes_used)
            target[0][target == 0] = 255
            target[0][target != 255] -= 1
        #
        return image, target

    def num_classes(self):
        nc = [self.num_classes_used]
        return nc

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.num_classes_used):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]
        #
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


###########################################
# config settings
def get_config(file_name):
    dataset_config = xnn.utils.ConfigNode()
    with open(file_name) as afp:
        datastore = json.load(afp)
    dataset_config.num_classes = len(datastore["categories"])
    return dataset_config


def write_to_jsonfile(path, filename, data):
    filepathnamewext = path + '/' + filename + '.json'
    if os.path.isfile(filepathnamewext) and os.access(filepathnamewext, os.R_OK):
        pass
    else:
        with open(filepathnamewext, 'a') as fp:
            json.dump(data, fp)


def common_segmentation(dataset_config, root, *args, split=None, transforms=None, annotation_prefix="stuff", with_background_class=False, **kwargs):
    train_split = val_split = None
    annotation_file = os.path.join(root, 'annotations', f'{annotation_prefix}.json')
    instances = dataset_split(annotation_file, 0.2)
    dataset_config = get_config(annotation_file).merge_from(dataset_config)

    split = ['train', 'val']
    for split_name in split:
        if split_name.startswith('train'):
            write_to_jsonfile(os.path.join(root, 'annotations'), f"{annotation_prefix}_{split_name}", instances[split_name])
            train_split = CommonSegmentationPlus(root, split_name, num_classes=dataset_config.num_classes,
                                                  transforms=transforms[0], annotation_prefix=annotation_prefix, with_background_class=with_background_class, *args, **kwargs)
        elif split_name.startswith('val'):
            write_to_jsonfile(os.path.join(root, 'annotations'), f"{annotation_prefix}_{split_name}", instances[split_name])
            val_split = CommonSegmentationPlus(root, split_name, num_classes=dataset_config.num_classes,
                                                transforms=transforms[1], annotation_prefix=annotation_prefix, with_background_class=with_background_class, *args, **kwargs)
        else:
            assert False, 'unknown split'
        #
    #
    return train_split, val_split


def common_segmentation_with_background_class(*args, with_background_class=True, **kwargs):
    return common_segmentation(*args, with_background_class=with_background_class, **kwargs)
