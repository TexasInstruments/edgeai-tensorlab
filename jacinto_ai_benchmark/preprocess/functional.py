import copy
import math
import numbers
import warnings
from typing import Any, Optional

import numpy as np
from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None

from . import functional_pil as F_pil


_is_pil_image = F_pil._is_pil_image
_parse_fill = F_pil._parse_fill


def _get_image_size(img):
    """Returns image sizea as (w, h)
    """

    return F_pil._get_image_size(img)


def _is_numpy(img: Any) -> bool:
    return isinstance(img, np.ndarray)


def _is_numpy_image(img: Any) -> bool:
    return img.ndim in {2, 3}


def to_numpy_tensor(img):
    """Convert a ``PIL Image`` to a tensor of the same type.

    See ``AsTensor`` for more details.

    Args:
        img (PIL Image): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if F_pil._is_pil_image(img):
        # handle PIL Image
        img = np.asarray(img)
    #
    if img.ndim == 2:
        img = img[:, :, None]
    #
    # put it from HWC to CHW format
    img = img.transpose((2, 0, 1))
    return img


def to_numpy_tensor_4d(img):
    img = to_numpy_tensor(img)
    if img.ndim == 3:
        img = img[None, :, :, :]
    #
    return img


def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """

    if not inplace:
        tensor = copy.deepcopy(tensor)

    mean = [mean] if isinstance(mean, numbers.Number) else mean
    std = [std] if isinstance(std, numbers.Number) else std

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion, leading to division by zero.')
    if mean.ndim == 1:
        mean = mean[:, None, None]
    #
    if std.ndim == 1:
        std = std[:, None, None]
    #
    if tensor.ndim < std.ndim:
        mean = mean[None, :, None, None]
        scale = std[None, :, None, None]
    #
    tensor = (tensor - mean) / (std)
    return tensor


def normalize_mean_scale(tensor, mean, scale, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        scale (sequence): Sequence of scaling values for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """

    if not inplace:
        tensor = copy.deepcopy(tensor)

    mean = [mean] if isinstance(mean, numbers.Number) else mean
    scale = [scale] if isinstance(scale, numbers.Number) else scale
    mean = np.array(mean, dtype=np.float32)
    scale = np.array(scale, dtype=np.float32)
    if mean.ndim == 1:
        mean = mean[:, None, None]
    #
    if scale.ndim == 1:
        scale = scale[:, None, None]
    #
    if tensor.ndim < scale.ndim:
        mean = mean[None, :, None, None]
        scale = scale[None, :, None, None]
    #
    tensor = (tensor - mean) * scale
    return tensor


def resize(img, size, interpolation: int = Image.BILINEAR):
    r"""Resize the input image to the given size.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        img (PIL Image or Tensor): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.
            In torchscript mode padding as single int is not supported, use a tuple or
            list of length 1: ``[size, ]``.
        interpolation (int, optional): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.

    Returns:
        PIL Image or Tensor: Resized image.
    """
    if _is_pil_image(img):
        return F_pil.resize(img, size=size, interpolation=interpolation)

    return F_np.resize(img, size=size, interpolation=interpolation)


def pad(img, padding, fill = 0, padding_mode = "constant"):
    r"""Pad the given image on all sides with the given "pad" value.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        img (PIL Image or Tensor): Image to be padded.
        padding (int or tuple or list): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.
            In torchscript mode padding as single int is not supported, use a tuple or
            list of length 1: ``[padding, ]``.
        fill (int or str or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant. Only int value is supported for Tensors.
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            Mode symmetric is not yet supported for Tensor inputs.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value on the edge of the image

            - reflect: pads with reflection of image (without repeating the last value on the edge)

                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image (repeating the last value on the edge)

                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        PIL Image or Tensor: Padded image.
    """
    if _is_pil_image(img):
        return F_pil.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)

    return F_np.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)


def crop(img, top, left, height, width):
    """Crop the given image at specified location and output size.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        img (PIL Image or Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        PIL Image or Tensor: Cropped image.
    """

    if _is_pil_image(img):
        return F_pil.crop(img, top, left, height, width)

    return F_np.crop(img, top, left, height, width)


def center_crop(img, output_size):
    """Crops the given image at the center.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int
            it is used for both directions.

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    image_width, image_height = _get_image_size(img)
    crop_height, crop_width = output_size

    # crop_top = int(round((image_height - crop_height) / 2.))
    # Result can be different between python func and scripted func
    # Temporary workaround:
    crop_top = int((image_height - crop_height + 1) * 0.5)
    # crop_left = int(round((image_width - crop_width) / 2.))
    # Result can be different between python func and scripted func
    # Temporary workaround:
    crop_left = int((image_width - crop_width + 1) * 0.5)
    return crop(img, crop_top, crop_left, crop_height, crop_width)


def resized_crop(img, top, left, height, width, size, interpolation = Image.BILINEAR):
    """Crop the given image and resize it to desired size.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.

    Args:
        img (PIL Image or Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        size (sequence or int): Desired output size. Same semantics as ``resize``.
        interpolation (int, optional): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
    Returns:
        PIL Image or Tensor: Cropped image.
    """
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
    return img
