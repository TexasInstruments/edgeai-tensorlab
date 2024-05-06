from .vision import VisionDataset
from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple, List
import random
import numpy as np
import json
import copy
import cv2

class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


class CocoCaptions(CocoDetection):
    """`MS Coco Captions <https://cocodataset.org/#captions-2015>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Example:

        .. code:: python

            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.ToTensor())

            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample

            print("Image Size: ", img.size())
            print(target)

        Output: ::

            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']

    """

    def _load_target(self, id: int) -> List[str]:
        return [ann["caption"] for ann in super()._load_target(id)]



class COCOSegmentation():
    '''
    Modified from torchvision: https://github.com/pytorch/vision/references/segmentation/coco_utils.py
    Reference: https://github.com/pytorch/vision/blob/master/docs/source/models.rst
    '''
    def __init__(self, root, split, shuffle=False, num_imgs=None, num_classes=None):
        from pycocotools.coco import COCO
        num_classes = 80 if num_classes is None else num_classes
        if num_classes == 21:
            self.categories = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
            self.class_names = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                                'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        else:
            self.categories = range(num_classes)
            self.class_names = None
        #

        dataset_folders = os.listdir(root)
        assert 'annotations' in dataset_folders, 'invalid path to coco dataset annotations'
        annotations_dir = os.path.join(root, 'annotations')

        image_base_dir = 'images' if ('images' in dataset_folders) else ''
        image_base_dir = os.path.join(root, image_base_dir)
        image_split_dirs = os.listdir(image_base_dir)
        image_dir = os.path.join(image_base_dir, split)

        self.coco_dataset = COCO(os.path.join(annotations_dir, f'instances_{split}.json'))

        self.cat_ids = self.coco_dataset.getCatIds()
        img_ids = self.coco_dataset.getImgIds()
        self.img_ids = self._remove_images_without_annotations(img_ids)

        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.img_ids)
        #

        if num_imgs is not None:
            self.img_ids = self.img_ids[:num_imgs]
            self.coco_dataset.imgs = {k:self.coco_dataset.imgs[k] for k in self.img_ids}
        #

        imgs = []
        for img_id in self.img_ids:
            img = self.coco_dataset.loadImgs([img_id])[0]
            imgs.append(os.path.join(image_dir, img['file_name']))
        #
        self.imgs = imgs
        self.num_imgs = len(self.imgs)

    def __getitem__(self, idx, with_label=True):
        if with_label:
            image = Image.open(self.imgs[idx])
            ann_ids = self.coco_dataset.getAnnIds(imgIds=self.img_ids[idx], iscrowd=None)
            anno = self.coco_dataset.loadAnns(ann_ids)
            image, anno = self._filter_and_remap_categories(image, anno)
            image, target = self._convert_polys_to_mask(image, anno)
            image = np.array(image)
            if image.ndim==2 or image.shape[2] == 1:
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
            ann_ids = self.coco_dataset.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco_dataset.loadAnns(ann_ids)
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
            obj["category_id"] = self.categories.index(obj["category_id"])
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
            target[masks.sum(0) > 1] = 255
        else:
            target = np.zeros((h, w), dtype=np.uint8)
        #
        return image, target

    def _convert_poly_to_mask(self, segmentations, height, width):
        from pycocotools import mask as coco_mask
        masks = []
        for polygons in segmentations:
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


class CocoClassification(VisionDataset):

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        with open(annFile) as ann_fp:
            self.dataset_store = json.load(ann_fp)
        #
        self.annotations_info = self._find_annotations_info()
        self.num_images = len(self.dataset_store['images'])
        self.classes = copy.deepcopy(self.dataset_store['categories'])
        class_ids = [class_info['id'] for class_info in self.classes]
        self.class_ids_min = min(class_ids)
        # if self.class_ids_min != 0:
        #     for class_id in range(self.class_ids_min):
        #         self.classes.insert(0, dict(id=class_id, name=f'background_{class_id}'))
        #     #
        # #

    def _find_annotations_info(self):
        image_id_to_file_id_dict = dict()
        file_id_to_image_id_dict = dict()
        annotations_info_list = []
        for file_id, image_info in enumerate(self.dataset_store['images']):
            image_id = image_info['id']
            image_id_to_file_id_dict[image_id] = file_id
            file_id_to_image_id_dict[file_id] = image_id
            annotations_info_list.append([])
        #
        for annotation_info in self.dataset_store['annotations']:
            if annotation_info:
                image_id = annotation_info['image_id']
                file_id = image_id_to_file_id_dict[image_id]
                annotations_info_list[file_id].append(annotation_info)
            #
        #
        return annotations_info_list

    def _load_image(self, id: int) -> Image.Image:
        path = self.dataset_store['images'][id]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        anno_info = self.annotations_info[id][0]
        category_id = anno_info["category_id"]
        category_id -= self.class_ids_min
        return category_id

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self._load_image(index)
        target = self._load_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return self.num_images

