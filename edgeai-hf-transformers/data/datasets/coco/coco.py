# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License"); # you may not use this file except in compliance with the License.
# You may obtain a copy of the License at #
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software # distributed under the License is distributed on an "AS IS" BASIS, # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and # limitations under the License.

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import datasets
from datasets.data_files import DataFilesDict 
from datasets.download.download_manager import ArchiveIterable, DownloadManager 
from datasets.features import Features 
from datasets.info import DatasetInfo
import os
import glob

# Typing
_TYPING_BOX = Tuple[float, float, float, float]

_CITATION = """\
@article{DBLP:journals/corr/LinMBHPRDZ14,
  author    = {Tsung{-}Yi Lin and
               Michael Maire and
               Serge J. Belongie and
               Lubomir D. Bourdev and
               Ross B. Girshick and
               James Hays and
               Pietro Perona and
               Deva Ramanan and
               Piotr Doll{\'{a}}r and
               C. Lawrence Zitnick},
  title     = {Microsoft {COCO:} Common Objects in Context},
  journal   = {CoRR},
  volume    = {abs/1405.0312},
  year      = {2014},
  url       = {http://arxiv.org/abs/1405.0312},
  archivePrefix = {arXiv},
  eprint    = {1405.0312},
  timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/LinMBHPRDZ14},
  bibsource = {dblp computer science bibliography, https://dblp.org} } """

_DESCRIPTION = """\
This dataset contains all COCO 2017 images and annotations split in training (118287 images) \
    and validation (5000 images).
"""

_HOMEPAGE = "https://cocodataset.org"

_URLS = {
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "train": "http://images.cocodataset.org/zips/train2017.zip",
    "val": "http://images.cocodataset.org/zips/val2017.zip",
}

_SPLITS = ["train", "validation"]

_PATHS = {
    "annotations": {
        "train": Path("annotations/instances_train2017.json"),
        "validation": Path("annotations/instances_val2017.json"),
    },
    "images": {
        "train": Path("train2017"),
        "validation": Path("val2017"),
    },
}

_CLASSES = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "street sign",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eye glasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "plate",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "mirror",
    "dining table",
    "window",
    "desk",
    "toilet",
    "door",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    # "hair brush",
]

def round_box_values(box, decimals=2):
    return [round(val, decimals) for val in box]

class COCOHelper:
    """Helper class to load COCO annotations"""

    def __init__(self, annotation_path: Path, images_dir: Path) -> None:
        with open(annotation_path, "r") as file:
            data = json.load(file)
        self.data = data
        
        dict_id2annot: Dict[int, Any] = {}
        for annot in self.annotations:
            dict_id2annot.setdefault(annot["image_id"], []).append(annot)

        # Sort by id
        dict_id2annot = {
            k: list(sorted(v, key=lambda a: a["id"])) for k, v in dict_id2annot.items()
        }

        self.dict_path2annot: Dict[str, Any] = {}
        self.dict_path2id: Dict[str, Any] = {}
        for img in self.images:
            path_img = os.path.join(images_dir, img["file_name"])
            idx = int(img["id"])
            annot = dict_id2annot.get(idx, [])
            self.dict_path2annot[path_img] = annot
            self.dict_path2id[path_img] = img["id"]


    def __len__(self) -> int:
        return len(self.data["images"])

    @property
    def info(self) -> Dict[str, Union[str, int]]:
        return self.data["info"]

    @property
    def licenses(self) -> List[Dict[str, Union[str, int]]]:
        return self.data["licenses"]

    @property
    def images(self) -> List[Dict[str, Union[str, int]]]:
        return self.data["images"]

    @property
    def annotations(self) -> List[Any]:
        return self.data["annotations"]

    @property
    def categories(self) -> List[Dict[str, Union[str, int]]]:
        return self.data["categories"]

    def get_annotations(self, image_path: str) -> List[Any]:
        return self.dict_path2annot.get(image_path, [])

    def get_image_id(self, image_path: str) -> int:
        return self.dict_path2id.get(image_path, -1)


class COCO2017(datasets.GeneratorBasedBuilder):
    """COCO 2017 dataset."""

    VERSION = datasets.Version("1.0.1")
    
    def _info(self) -> datasets.DatasetInfo:
        """
        Returns the dataset metadata and features.

        Returns:
            DatasetInfo: Metadata and features of the dataset.
        """
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "image_id": datasets.Value("int64"),
                    "objects": datasets.Sequence(
                        {
                            "id": datasets.Value("int64"),
                            "area": datasets.Value("float64"),
                            "bbox": datasets.Sequence(
                                datasets.Value("float32"), length=4
                            ),
                            "category": datasets.ClassLabel(names=_CLASSES),
                            "iscrowd": datasets.Value("bool"),
                        }
                    ),
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """
        Provides the split information and downloads the data.

        Args:
            dl_manager (DownloadManager): The DownloadManager to use for downloading and
                extracting data.

        Returns:
            List[SplitGenerator]: List of SplitGenerator objects representing the data splits.
        """
        # archive_annots = dl_manager.download_and_extract(_URLS["annotations"])
        
        data_dir = self.config.data_dir
        if not data_dir:
            raise ValueError(
                "This script is supposed to work with local (downloaded) COCO dataset. The argument `data_dir` in `load_dataset()` is required."
            )

        splits = []
        for split in _SPLITS:
            # archive_split = dl_manager.download(_URLS[split])
            annotation_path = os.path.join(data_dir, _PATHS["annotations"][split])
            # images = dl_manager.iter_archive(archive_split)
            images = glob.glob(os.path.join(data_dir, _PATHS["images"][split], '*.jpg'))
            images_dir = os.path.join(data_dir, _PATHS["images"][split])
            
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split(split),
                    gen_kwargs={
                        "annotation_path": annotation_path,
                        "images_dir": images_dir,
                        "images": images
                    },
                )
            )
        return splits
            
    def _generate_examples(
        self, annotation_path: Path, images_dir: Path, images: List
    ) -> Iterator:
        """
        Generates examples for the dataset.

        Args:
            annotation_path (Path): The path to the annotation file.
            images_dir (Path): The path to the directory containing the images.
            images: (ArchiveIterable): An iterable containing the images.

        Yields:
            Dict[str, Union[str, Image]]: A dictionary containing the generated examples.
        """
        coco_annotation = COCOHelper(annotation_path, images_dir)
        
        for image_path in images:
            annotations = coco_annotation.get_annotations(image_path)
            with open(image_path, "rb") as fh:
                ret = {
                    "image": {"path": image_path, "bytes": fh.read()},
                    # "image": fh.read(),
                    "image_id": coco_annotation.get_image_id(image_path),
                    "objects": [
                        {
                            "id": annot["id"],
                            "area": annot["area"],
                            "bbox": round_box_values(annot["bbox"], 2), # [x, y, w, h]
                            "category": annot["category_id"],
                            "iscrowd": bool(annot["iscrowd"]),
                        }
                        for annot in annotations
                    ],
                }

            yield image_path, ret
