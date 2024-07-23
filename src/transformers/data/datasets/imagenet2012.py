import os
import glob
from io import BytesIO
import datasets
from .classes import IMAGENET2012_CLASSES


_CITATION = """\
@article{imagenet15russakovsky,
    Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
    Title = { {ImageNet Large Scale Visual Recognition Challenge} },
    Year = {2015},
    journal   = {International Journal of Computer Vision (IJCV)},
    doi = {10.1007/s11263-015-0816-y},
    volume={115},
    number={3},
    pages={211-252}
}
"""

_HOMEPAGE = "https://image-net.org/index.php"

_DESCRIPTION = """\
ILSVRC 2012, commonly known as 'ImageNet' is an image dataset organized according to the WordNet hierarchy. Each meaningful concept in WordNet, possibly described by multiple words or word phrases, is called a "synonym set" or "synset". There are more than 100,000 synsets in WordNet, majority of them are nouns (80,000+). ImageNet aims to provide on average 1000 images to illustrate each synset. Images of each concept are quality-controlled and human-annotated. In its completion, ImageNet hopes to offer tens of millions of cleanly sorted images for most of the concepts in the WordNet hierarchy. ImageNet 2012 is the most commonly used subset of ImageNet. This dataset spans 1000 object classes and contains 1,281,167 training images, 50,000 validation images and 100,000 test images
"""


class ImageNet1K(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    DEFAULT_WRITER_BATCH_SIZE = 1000

    def _info(self):
        assert len(IMAGENET2012_CLASSES) == 1000
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=list(IMAGENET2012_CLASSES.values())),
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        split_names = ['train', 'val']
        base_path = os.path.join(kwargs.get('base_path', ''), 'data')
        self.split_folders = {split_name: os.path.join(base_path, split_name) for split_name in split_names}
    
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # iglob will return the iterator, so sharding will not be supported, gen_kwargs should pass something what can be broken down to lists
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filename_iterator": glob.glob(self.split_folders['train'] + '/*/*.JPEG'),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filename_iterator": glob.glob(self.split_folders['val'] + '/*/*.JPEG'),
                    "split": "val",
                },
            )
        ]   
    
    def _generate_examples(self, filename_iterator, split):
        for idx, filename in enumerate(filename_iterator):
            if split in filename and filename.endswith(".JPEG"):
                with open(filename, "rb") as fh:
                    buf =  BytesIO(fh.read())
                    yield idx, {
                        "image": {"path": filename, "bytes": buf},
                        "label": IMAGENET2012_CLASSES[os.path.basename(os.path.dirname(filename))]
                    }