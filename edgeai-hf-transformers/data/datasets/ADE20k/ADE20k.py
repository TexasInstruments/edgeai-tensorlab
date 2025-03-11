import datasets
import os
from pathlib import Path
import glob

_VERSION = datasets.Version("1.0.1")

_DESCRIPTION = "This dataset contains all the ADE20k images and annotation images split in training and validation."
_HOMEPAGE = "https://groups.csail.mit.edu/vision/datasets/ADE20K/"
_LICENSE = "Creative Commons BSD-3 License Agreement"
_CITATION = " @article{zhou2019semantic, \
  title={Semantic understanding of scenes through the ade20k dataset}, \
  author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Xiao, Tete and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio}, \
  journal={International Journal of Computer Vision}, \
  volume={127}, \
  number={3}, \
  pages={302--321}, \
  year={2019}, \
  publisher={Springer}"

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "label": datasets.Image(),
    },
)

_SPLITS = ["train", "validation"]

_PATHS = {
    "annotations": {
        "train": Path("annotations/training"),
        "validation": Path("annotations/validation"),
    },
    "images": {
        "train": Path("images/training"),
        "validation": Path("images/validation"),
    },
}

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)


class ADE20k(datasets.GeneratorBasedBuilder):
    
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        
        data_dir = self.config.data_dir
        if not data_dir:
            raise ValueError(
                "This script is supposed to work with local (downloaded) ADE20K dataset. The argument `data_dir` in `load_dataset()` is required."
            )
            
        splits = []
        for split in _SPLITS:
            images = glob.glob(os.path.join(data_dir, _PATHS["images"][split], '*.jpg'))
            images_dir = os.path.join(data_dir, _PATHS["images"][split])
            annotations_dir = os.path.join(data_dir, _PATHS["annotations"][split])
            
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split(split),
                    gen_kwargs={
                        "annotations_dir": annotations_dir,
                        "images": images,
                        "split" : split
                    },
                )
            )
        return splits
    

    def _generate_examples(self, annotations_dir, images, split):
        for image_path in images:
            ann_img = os.path.join(annotations_dir, image_path.split("/")[-1])[:-4] + ".png" 
            # if split=="validation":
            #     fh = open(image_path, "rb")
            #     ret = {
            #         "image" : {"path": image_path, "bytes": fh.read()},
            #         "label" : {"path": ann_img, "bytes": None}
            #     }
            #     fh.close()
            #     yield image_path, ret
            # else:
            fh = open(image_path, "rb")
            fh_annotation = open(ann_img, "rb") 
            ret = {
                "image" : {"path": image_path, "bytes": fh.read()},
                "label" : {"path": ann_img, "bytes": fh_annotation.read()}
            }
            fh.close()
            fh_annotation.close()
            yield image_path, ret