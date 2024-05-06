# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from .. import constants
from .. import utils
from .transforms import *


class PreProcessTransforms:
    def __init__(self, settings):
        self.settings = settings

    ###############################################################
    # preprocess transforms
    ###############################################################
    def get_transform_base(self, resize, crop, data_layout, reverse_channels,
                         backend, interpolation, resize_with_pad, mean, scale,
                         add_flip_image=False, pad_color=0):
        transforms_list = [
            ImageRead(backend=backend),
            ImageResize(resize, interpolation=interpolation, resize_with_pad=resize_with_pad, pad_color=pad_color),
            ImageCenterCrop(crop),
            ImageToNPTensor4D(data_layout=data_layout),
            ImageNormMeanScale(mean=mean, scale=scale, data_layout=data_layout)]
        if reverse_channels:
            transforms_list = transforms_list + [NPTensor4DChanReverse(data_layout=data_layout)]
        if add_flip_image:
            transforms_list += [ImageFlipAdd()]
        #
        transforms = utils.TransformsCompose(transforms_list, resize=resize, crop=crop,
                                             data_layout=data_layout, reverse_channels=reverse_channels,
                                             backend=backend, interpolation=interpolation,
                                             mean=mean, scale=scale, add_flip_image=add_flip_image)
        return transforms

    def get_transform_onnx(self, resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
                         backend='pil', interpolation=None, resize_with_pad=False,
                         mean=(123.675, 116.28, 103.53), scale=(0.017125, 0.017507, 0.017429), add_flip_image=False, pad_color=0):
        transforms = self.get_transform_base(resize=resize, crop=crop, data_layout=data_layout,
                                      reverse_channels=reverse_channels, backend=backend, interpolation=interpolation,
                                      resize_with_pad=resize_with_pad, mean=mean, scale=scale, add_flip_image=add_flip_image, pad_color=pad_color)
        return transforms

    def get_transform_jai(self, resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False,
                        mean=(128.0, 128.0, 128.0), scale=(1/64.0, 1/64.0, 1/64.0)):
        return self.get_transform_base(resize=resize, crop=crop, data_layout=data_layout, reverse_channels=reverse_channels,
                                backend=backend, interpolation=interpolation, resize_with_pad=resize_with_pad,
                                mean=mean, scale=scale)

    def get_transform_mxnet(self, resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=None, resize_with_pad=False,
                        mean=(123.675, 116.28, 103.53), scale=(0.017125, 0.017507, 0.017429)):
        return self.get_transform_base(resize=resize, crop=crop, data_layout=data_layout, reverse_channels=reverse_channels,
                                backend=backend, interpolation=interpolation, resize_with_pad=resize_with_pad,
                                mean=mean, scale=scale)

    def get_transform_tflite(self, resize=256, crop=224, data_layout=constants.NHWC, reverse_channels=False,
                              backend='pil', interpolation=None, resize_with_pad=False,
                              mean=(128.0, 128.0, 128.0), scale=(1/128.0, 1/128.0, 1/128.0)):
        return self.get_transform_base(resize=resize, crop=crop, data_layout=data_layout, reverse_channels=reverse_channels,
                                backend=backend, interpolation=interpolation, resize_with_pad=resize_with_pad,
                                mean=mean, scale=scale)

    def get_transform_tflite_quant(self, *args, mean=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), **kwargs):
        return self.get_transform_tflite(*args, mean=mean, scale=scale, **kwargs)
