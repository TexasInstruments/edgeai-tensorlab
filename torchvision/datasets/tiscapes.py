import json
import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from .utils import  verify_str_arg, download_and_extract_archive
from .vision import VisionDataset
from PIL import Image


class TIscapes(VisionDataset):
    """`TIscapes <http://www.TIscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory `images` are located.
        mode (string, optional): The quality mode to use, ``fine``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = TIscapes('./data/TIscapes', mode='fine')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = TIscapes('./data/TIscapes', mode='fine',
                                 target_type=['color'])

            img, (col, poly) = dataset[0]
    """

    def __init__(
            self,
            root: str,
            split: str = "",
            mode: str = "fine",
            target_type: Union[List[str], str] = "color",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(TIscapes, self).__init__(root, transforms, transform, target_transform)
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(os.path.dirname(self.root), 'images', split)
        self.targets_dir = os.path.join(os.path.dirname(self.root), 'polygons', split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [verify_str_arg(value, "target_type",
                        ("polygon", "color"))
         for value in self.target_type]

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):

            download_and_extract_archive(
                "http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/datasets/tiscapes2017_driving.zip",
                os.path.dirname(self.root),
                os.path.dirname(self.root),
            )

        for image in os.listdir(self.images_dir):
            target_types = []
            for t in self.target_type:
                target_name = '{}_{}'.format(image.split('.jpg')[0],
                                             self._get_target_suffix(t))
                target_types.append(os.path.join(self.targets_dir, target_name))

            self.images.append(os.path.join(self.images_dir, image))
            self.targets.append(target_types)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, target_type: str) -> str:
        if target_type == 'color':
            return 'gtFine_color.png'
        else:
            return 'gtFine_polygons.json'
