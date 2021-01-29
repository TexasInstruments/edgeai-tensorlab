import numpy as np
import PIL
from PIL import ImageOps
import cv2


def get_size(input):
    size = input.shape[:2][::-1] if isinstance(input, np.ndarray) else input.size
    return size


def get_resize_dimensions(inResizeType, image_w, image_h, resize_w, resize_h):
    if inResizeType == 0:
        pass
    elif inResizeType == 1 or inResizeType == 3:
        # resize min size to fit the given size
        # max size may exceed the given size
        # inResizeType == 3 will not do cropping if the size exceeds the size
        if image_h < image_w:
            resize_w = round(resize_h * (float(image_w) / float(image_h)))
        else:
            resize_h = round(resize_w * (float(image_h) / float(image_w)))
        #
    elif inResizeType == 2:
        # resize max size to fit the given size
        # min size may be smaller that the given size and may need padding.
        resize_w_orig, resize_h_orig = resize_w, resize_h
        resize_h = round(resize_w * (float(image_h) / float(image_w)))
        if resize_h > resize_h_orig:
            resize_h = round(resize_h * (float(resize_h_orig) / float(resize_h)))
        #
        resize_w = round(resize_h * (float(image_w) / float(image_h)))
        resize_w = min(resize_w, resize_w_orig)
        resize_h = min(resize_h, resize_h_orig)
    #
    return resize_w, resize_h


def resize_image(image, resize_w, resize_h, inResizeType=0, resample_type=None, resample_backend=None):
    assert resample_type is not None, 'resample_type must be provided'
    if resample_backend == 'pillow':
        image = PIL.Image.fromarray(image) if isinstance(input, np.ndarray) else image
    elif resample_backend == 'cv2':
        image = np.array(image)
    #
    image_w, image_h = get_size(image)
    resize_w, resize_h = get_resize_dimensions(inResizeType, image_w, image_h, resize_w, resize_h)
    if isinstance(image, np.ndarray):
        image = cv2.resize(image, dsize=(resize_w, resize_h), interpolation=resample_type)
    else:
        image = image.resize((resize_w, resize_h), resample=resample_type)
    #
    return image


def pad_image(image, size):
    if image.size == size:
        return image
    #
    if isinstance(image, np.ndarray):
        image_w, image_h = image.shape[:2][::-1]
        left = (size[0] - image_w)//2
        right = (size[0] - image_w)
        top = (size[1] - image_h)//2
        bottom = (size[1] - image_h)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    else:
        image = ImageOps.pad(image, size)
    #
    return image


def resize_pad_crop_image(input, resize_w=None, resize_h=None, crop_w=None, crop_h=None,
                          inResizeType=0, resample_type=None, resample_backend=None):
    if resample_backend == 'pillow':
        input = PIL.Image.fromarray(input) if isinstance(input, np.ndarray) else input
    elif resample_backend == 'cv2':
        input = np.array(input)
    #
    if resize_w is not None and resize_h is not None:
        input = resize_image(input, resize_w, resize_h, inResizeType=inResizeType, resample_type=resample_type,
                             resample_backend=resample_backend)
        input_resized_w, input_resized_h = get_size(input)
    else:
        input_resized_w, input_resized_h = resize_w, resize_h
    #
    if inResizeType == 1 or inResizeType == 2:
        if (input_resized_w<resize_w or input_resized_h<resize_h):
            input = pad_image(input, size=(resize_w,resize_h))
            input_resized_w, input_resized_h = get_size(input)
        #
        if crop_w is not None and crop_h is not None:
            crop_offset_w = (input_resized_w - crop_w) // 2
            crop_offset_h = (input_resized_h - crop_h) // 2
            if isinstance(input, np.ndarray):
                input = input[crop_offset_h:(crop_offset_h+crop_h), crop_offset_w:(crop_offset_w+crop_w), ...]
            else:
                input = input.crop((crop_offset_w,crop_offset_h,crop_offset_w+crop_w,crop_offset_h+crop_h))
            #
        #
    #
    return input

