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
from .bev_detection import *


class PreProcessTransforms(utils.TransformsCompose):
    def __init__(self, settings, transforms=None, **kwargs):
        super().__init__(transforms, **kwargs)
        self.settings = settings

    def set_input_size(self, resize, crop):
        for t in self.transforms:
            if isinstance(t, ImageResize):
                t.set_size(resize)
            elif isinstance(t, ImageCenterCrop):
                t.set_size(crop)
            #
        #

    ###############################################################
    # preprocess transforms
    ###############################################################
    def get_transform_base(self, resize, crop, data_layout, reverse_channels,
                         backend, interpolation, resize_with_pad,
                         add_flip_image=False, pad_color=0):
        if resize is None:
            transforms_list = [
                ImageRead(backend=backend),
                ImageCenterCrop(crop),
                ImageToNPTensor4D(data_layout=data_layout)
            ]
        else:
            transforms_list = [
                ImageRead(backend=backend),
                ImageResize(resize, interpolation=interpolation, resize_with_pad=resize_with_pad, pad_color=pad_color),
                ImageCenterCrop(crop),
                ImageToNPTensor4D(data_layout=data_layout)
            ]

        if reverse_channels:
            transforms_list = transforms_list + [NPTensor4DChanReverse(data_layout=data_layout)]
        if add_flip_image:
            transforms_list += [ImageFlipAdd()]
        #
        transforms = PreProcessTransforms(None, transforms_list,
                                          resize=resize, crop=crop,
                                          data_layout=data_layout, reverse_channels=reverse_channels,
                                          backend=backend, interpolation=interpolation,
                                          add_flip_image=add_flip_image, resize_with_pad=resize_with_pad, pad_color=pad_color)
        return transforms

    def get_transform_onnx(self, resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
                         backend='pil', interpolation=None, resize_with_pad=False,
                         add_flip_image=False, pad_color=0):
        transforms = self.get_transform_base(resize=resize, crop=crop, data_layout=data_layout,
                                      reverse_channels=reverse_channels, backend=backend, interpolation=interpolation,
                                      resize_with_pad=resize_with_pad, add_flip_image=add_flip_image, pad_color=pad_color)
        return transforms

    def get_transform_jai(self, resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False):
        return self.get_transform_base(resize=resize, crop=crop, data_layout=data_layout, reverse_channels=reverse_channels,
                                backend=backend, interpolation=interpolation, resize_with_pad=resize_with_pad)

    def get_transform_mxnet(self, resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=None, resize_with_pad=False):
        return self.get_transform_base(resize=resize, crop=crop, data_layout=data_layout, reverse_channels=reverse_channels,
                                backend=backend, interpolation=interpolation, resize_with_pad=resize_with_pad)

    def get_transform_tflite(self, resize=256, crop=224, data_layout=constants.NHWC, reverse_channels=False,
                              backend='pil', interpolation=None, resize_with_pad=False, pad_color=0):
        return self.get_transform_base(resize=resize, crop=crop, data_layout=data_layout, reverse_channels=reverse_channels,
                                backend=backend, interpolation=interpolation, resize_with_pad=resize_with_pad,
                                pad_color=pad_color)

    def get_transform_tflite_quant(self, *args, **kwargs):
        return self.get_transform_tflite(*args, **kwargs)

    def get_transform_lidar_base(self):
        transforms_list = [
            PointCloudRead(),
            Voxelization()
            ]
        transforms = PreProcessTransforms(None, transforms_list)

        return transforms

    """
    def get_bev_base_transform(self, imsize=256, resize=256, crop=224, data_layout=constants.NCHW, 
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False, pad_color=0):
        transforms_list = [
            BEVSensorsRead(imsize, resize, crop),
            ImageRead(backend=backend, bgr_to_rgb=False),
            ImageResize(resize, interpolation=interpolation, resize_with_pad=resize_with_pad, pad_color=pad_color),
            ImageCrop(crop),
            ImageToNPTensor4D(data_layout=data_layout),
        ]

        return transforms_list
    """

    def get_transform_bev_petr(self, imsize=256, resize=256, crop=224, featsize=(20, 50), data_layout=constants.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False, pad_color=0):
        transforms_list = [
            BEVSensorsRead(imsize, resize, crop),
            ImageRead(backend=backend, bgr_to_rgb=False),
            ImageResize(resize, interpolation=interpolation, resize_with_pad=resize_with_pad, pad_color=pad_color),
            ImageCrop(crop),
            ImageToNPTensor4D(data_layout=data_layout),
            GetPETRGeometry(crop, featsize)
        ]

        transforms = PreProcessTransforms(None, transforms_list,
                                          imsize=imsize, resize=resize, crop=crop,
                                          data_layout=data_layout, reverse_channels=reverse_channels,
                                          backend=backend, interpolation=interpolation,
                                          resize_with_pad=resize_with_pad, pad_color=pad_color)
        return transforms


    def get_transform_bev_bevdet(self, imsize=256, resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False, pad_color=0):
        transforms_list = [
            BEVSensorsRead(imsize, resize, crop),
            ImageRead(backend=backend, bgr_to_rgb=False),
            ImageResize(resize, interpolation=interpolation, resize_with_pad=resize_with_pad, pad_color=pad_color),
            ImageCrop(crop),
            ImageToNPTensor4D(data_layout=data_layout),
            GetBEVDetGeometry(crop)
        ]

        transforms = PreProcessTransforms(None, transforms_list,
                                          imsize=imsize, resize=resize, crop=crop,
                                          data_layout=data_layout, reverse_channels=reverse_channels,
                                          backend=backend, interpolation=interpolation,
                                          resize_with_pad=resize_with_pad, pad_color=pad_color)
        return transforms


    def get_transform_bev_bevformer(self, imsize=256, resize=256, pad=224, data_layout=constants.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False, pad_color=0):
        transforms_list = [
            BEVSensorsRead(imsize, resize, (0, 0, resize[1]+pad[2], resize[0]+pad[3])),
            ImageRead(backend=backend, bgr_to_rgb=True),
            ImageResize(resize, interpolation=interpolation, resize_with_pad=resize_with_pad, pad_color=pad_color),
            ImagePad(pad),
            ImageToNPTensor4D(data_layout=data_layout),
            GetBEVFormerGeometry(pad)
        ]

        transforms = PreProcessTransforms(None, transforms_list,
                                          imsize=imsize, resize=resize, pad=pad,
                                          data_layout=data_layout, reverse_channels=reverse_channels,
                                          backend=backend, interpolation=interpolation,
                                          resize_with_pad=resize_with_pad, pad_color=pad_color)
        return transforms


    def get_transform_fcos3d(self, imsize=256, resize=256, pad=224, data_layout=constants.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False, pad_color=0):

        transforms_list = [
            BEVSensorsRead(imsize, resize, (0, 0, resize[1]+pad[2], resize[0]+pad[3]), load_type='mv_image_based'),
            ImageRead(backend=backend, bgr_to_rgb=False),
            ImagePad(pad),
            ImageToNPTensor4D(data_layout=data_layout),
            GetFCOS3DGeometry()
        ]

        transforms = PreProcessTransforms(None, transforms_list,
                                          imsize=imsize, resize=resize, pad=pad,
                                          data_layout=data_layout, reverse_channels=reverse_channels,
                                          backend=backend, interpolation=interpolation,
                                          resize_with_pad=resize_with_pad, pad_color=pad_color)
        return transforms



    def get_transform_bev_fastbev(self, imsize=256, resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False, pad_color=0):
        transforms_list = [
            BEVSensorsRead(imsize, resize, crop),
            ImageRead(backend=backend, bgr_to_rgb=True),
            ImageResize(resize, interpolation=interpolation, resize_with_pad=resize_with_pad, pad_color=pad_color),
            ImageCrop(crop),
            ImageToNPTensor4D(data_layout=data_layout),
            GetFastBEVGeometry(crop)
        ]

        transforms = PreProcessTransforms(None, transforms_list,
                                          imsize=imsize, resize=resize, crop=crop,
                                          data_layout=data_layout, reverse_channels=reverse_channels,
                                          backend=backend, interpolation=interpolation,
                                          resize_with_pad=resize_with_pad, pad_color=pad_color)
        return transforms


    def get_transform_none(self):
        return PreProcessTransforms(self.settings, transforms=[])

