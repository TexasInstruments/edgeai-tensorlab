import numpy as np
import cv2
from typing import Any, List, Sequence


def crop(img, top: int, left, height, width):
    """Crop the given np array.

    Args:
        img (np.ndarray): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        np.ndarray: Cropped image.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img[top:(top + height), left:(left + width), ...]


def resize(img, size, **kwargs):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        dsize (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.
            For compatibility reasons with ``functional_tensor.resize``, if a tuple or list of length 1 is provided,
            it is interpreted as a single int.
        interpolation (int, optional): Desired interpolation. Default is ``PIL.Image.BILINEAR``.

    Returns:
        PIL Image: Resized image.
    """
    interpolation = kwargs.get('interpolation', None) or cv2.INTER_LINEAR

    if not isinstance(img, np.ndarray):
        raise TypeError('img should be numpy array. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Sequence) and len(size) in (1, 2))):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int) or len(size) == 1:
        if isinstance(size, Sequence):
            size = size[0]
        w, h = img.shape[1], img.shape[0]
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(img, (ow, oh), interpolation=interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(img, (ow, oh), interpolation=interpolation)
    else:
        return cv2.resize(img, dsize=size[::-1], interpolation=interpolation)