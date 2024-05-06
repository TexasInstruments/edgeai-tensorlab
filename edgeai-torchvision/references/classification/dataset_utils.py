from torchvision.datasets import VisionDataset, DatasetFolder
from torchvision.datasets.folder  import default_loader, IMG_EXTENSIONS
from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple, List, Dict
import random
import numpy as np
import json
import copy
import cv2


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



###############################################################
class DataListClassification(DatasetFolder):
    def __init__(
            self,
            files_path: str,
            split_file: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(files_path, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self.find_classes(files_path, split_file)
        samples = self.make_dataset(files_path, split_file, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def find_classes(self, files_path: str, split_file: str) -> Tuple[List[str], Dict[str, int]]:
        with open(split_file) as sfp:
            split_lines = sfp.readlines()
        #
        split_lines = [split_line.strip().split(' ') for split_line in split_lines]
        split_labels = [split_line[1] for split_line in split_lines]
        num_class = max(split_labels)+1
        classes = range(num_class)
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(self, files_path: str, split_file: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        with open(split_file) as sfp:
            instances = sfp.readlines()
            instances = [split_line.strip().split(' ') for split_line in split_lines]
            instances = [split_line for split_line in split_lines if is_valid_file(split_line[0])]
        #
        return split_lines


class ImageListClassification(DataListClassification):
    def __init__(
            self,
            files_path: str,
            split_file,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(files_path, split_file, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
