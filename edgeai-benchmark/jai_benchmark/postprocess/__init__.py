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


class PostProcessTransforms:
    def __init__(self, settings):
        self.settings = settings

    ###############################################################
    # post process transforms for classification
    ###############################################################
    def get_transform_classification(self):
        postprocess_classification = [IndexArray(), ArgMax()]
        transforms = utils.TransformsCompose(postprocess_classification)
        return transforms

    ###############################################################
    # post process transforms for detection
    ###############################################################
    def get_transform_detection_base(self, formatter=None, resize_with_pad=False, normalized_detections=True,
                                     shuffle_indices=None, squeeze_axis=0, reshape_list=None, ignore_detection_element=None):
        postprocess_detection = [ReshapeList(reshape_list=reshape_list),
                                 ShuffleList(indices=shuffle_indices),
                                 Concat(axis=-1, end_index=3)]
        if squeeze_axis is not None:
            #  TODO make this more generic to squeeze any axis
            postprocess_detection += [IndexArray()]
        #
        if ignore_detection_element is not None:
            postprocess_detection += [IgnoreDetectionElement(ignore_detection_element)]
        #
        if formatter is not None:
            postprocess_detection += [formatter]
        #
        postprocess_detection += [DetectionResizePad(resize_with_pad=resize_with_pad,
                                                    normalized_detections=normalized_detections)]
        if self.settings.detection_thr is not None:
            postprocess_detection += [DetectionFilter(detection_thr=self.settings.detection_thr,
                                                                  detection_max=self.settings.detection_max)]
        #
        if self.settings.save_output:
            postprocess_detection += [DetectionImageSave()]
        #
        transforms = utils.TransformsCompose(postprocess_detection, detection_thr=self.settings.detection_thr,
                            save_output=self.settings.save_output, formatter=formatter, resize_with_pad=resize_with_pad,
                            normalized_detections=normalized_detections, shuffle_indices=shuffle_indices,
                            squeeze_axis=squeeze_axis)
        return transforms

    def get_transform_detection_onnx(self, formatter=None, **kwargs):
        return self.get_transform_detection_base(formatter=formatter, **kwargs)

    def get_transform_detection_mmdet_onnx(self, formatter=None, **kwargs):
        return self.get_transform_detection_base(formatter=formatter, reshape_list=[(-1,5), (-1,1)], **kwargs)

    def get_transform_detection_yolov5_onnx(self, formatter=None, **kwargs):
        return self.get_transform_detection_base(formatter=formatter, reshape_list=[(-1,6)], **kwargs)

    def get_transform_detection_tflite(self, formatter=DetectionYXYX2XYXY(), **kwargs):
        return self.get_transform_detection_base(formatter=formatter, **kwargs)

    def get_transform_detection_mxnet(self, formatter=None, resize_with_pad=False,
                        normalized_detections=False, shuffle_indices=(2,0,1), **kwargs):
        return self.get_transform_detection_base(formatter=formatter, resize_with_pad=resize_with_pad,
                        normalized_detections=normalized_detections, shuffle_indices=shuffle_indices, **kwargs)

    ###############################################################
    # post process transforms for segmentation
    ###############################################################
    def get_transform_segmentation_base(self, data_layout, with_argmax=True):
        channel_axis = -1 if data_layout == constants.NHWC else 1
        postprocess_segmentation = [IndexArray()]
        if with_argmax:
            postprocess_segmentation += [ArgMax(axis=channel_axis)]
        #
        postprocess_segmentation += [NPTensorToImage(data_layout=data_layout),
                                     SegmentationImageResize(),
                                     SegmentationImagetoBytes()]
        if self.settings.save_output:
            postprocess_segmentation += [SegmentationImageSave()]
        #
        transforms = utils.TransformsCompose(postprocess_segmentation, data_layout=data_layout,
                                             save_output=self.settings.save_output, with_argmax=with_argmax)
        return transforms

    def get_transform_segmentation_onnx(self, data_layout=constants.NCHW, with_argmax=True):
        return self.get_transform_segmentation_base(data_layout=data_layout, with_argmax=with_argmax)

    def get_transform_segmentation_tflite(self, data_layout=constants.NHWC, with_argmax=True):
        return self.get_transform_segmentation_base(data_layout=data_layout, with_argmax=with_argmax)

    ###############################################################
    # post process transforms for human pose estimation
    ###############################################################
    def get_transform_human_pose_estimation_base(self, data_layout, with_udp=True):
        # channel_axis = -1 if data_layout == constants.NHWC else 1
        # postprocess_human_pose_estimation = [IndexArray()] #just removes the first axis from output list, final size (c,w,h)
        postprocess_human_pose_estimation = [HumanPoseHeatmapParser(use_udp=with_udp),
                                             KeypointsProject2Image(use_udp=with_udp)]

        if self.settings.save_output:
            postprocess_human_pose_estimation += [HumanPoseImageSave()]
        #
        transforms = utils.TransformsCompose(postprocess_human_pose_estimation, data_layout=data_layout,
                                             save_output=self.settings.save_output)
        return transforms

    def get_transform_human_pose_estimation_onnx(self, data_layout=constants.NCHW):
        return self.get_transform_human_pose_estimation_base(data_layout=data_layout, with_udp=self.settings.with_udp)
