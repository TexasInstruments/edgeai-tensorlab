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

from ..common.postprocess_utils import *
from .. import constants
from .. import utils
from .transforms import *
from .keypoints import *
from .object_6d_pose import *
from . import transforms as postprocess_transform_types

class PostProcessTransforms(utils.TransformsCompose):
    def __init__(self, settings, transforms=None, **kwargs):
        super().__init__(transforms, **kwargs)
        self.settings = settings

    ###############################################################
    # post process transforms for classification
    ###############################################################
    def get_transform_none(self, **kwargs):
        postprocess_none = []
        transforms = PostProcessTransforms(None, postprocess_none)
        return transforms

    ###############################################################
    # post process transforms for classification
    ###############################################################
    def get_transform_classification(self):
        postprocess_classification = [SqueezeAxis(), ArgMax(axis=-1)]
        if self.settings.save_output:
            postprocess_classification += [ClassificationImageSave(self.settings.num_output_frames)]
        #
        transforms = PostProcessTransforms(None, postprocess_classification)
        return transforms

    ###############################################################
    # post process transforms for detection
    ###############################################################
    def get_transform_detection_base(self, formatter=None, resize_with_pad=False, keypoint=False, object6dpose=False, normalized_detections=True,
                                     shuffle_indices=None, squeeze_axis=0, reshape_list=None, ignore_index=None, logits_bbox_to_bbox_ls=False):
        
        postprocess_detection = []
        if logits_bbox_to_bbox_ls:
            postprocess_detection += [LogitsToLabelScore()]
        #
        postprocess_detection += [ReshapeList(reshape_list=reshape_list),
                                 ShuffleList(indices=shuffle_indices),
                                 Concat(axis=-1, end_index=3)]
        if squeeze_axis is not None:
            #  TODO make this more generic to squeeze any axis
            postprocess_detection += [SqueezeAxis()]
        #
        if ignore_index is not None:
            postprocess_detection += [IgnoreIndex(ignore_index)]
        #
        if formatter is not None:
            if isinstance(formatter, str):
                formatter_name = formatter
                formatter = getattr(postprocess_transform_types, formatter_name)()
            elif isinstance(formatter, dict) and 'type' in formatter:
                formatter_name = formatter.pop('type')
                formatter = getattr(postprocess_transform_types, formatter_name)(**formatter)
            #
            postprocess_detection += [formatter]
        #
        postprocess_detection += [DetectionResizePad(resize_with_pad=resize_with_pad, keypoint=keypoint, object6dpose=object6dpose,
                                                    normalized_detections=normalized_detections)]
        if self.settings.detection_threshold is not None:
            postprocess_detection += [DetectionFilter(detection_threshold=self.settings.detection_threshold,
                                                                  detection_keep_top_k=self.settings.detection_keep_top_k)]
        #
        if keypoint:
            postprocess_detection += [BboxKeypointsConfReformat()]
        if object6dpose:
            postprocess_detection += [BboxObject6dPoseReformat()]

        if self.settings.save_output:
            if keypoint:
                postprocess_detection += [HumanPoseImageSave(self.settings.num_output_frames)]
            elif object6dpose:
                postprocess_detection += [Object6dPoseImageSave(self.settings.num_output_frames)]
            else:
                postprocess_detection += [DetectionImageSave(self.settings.num_output_frames)]
        #
        transforms = PostProcessTransforms(None, postprocess_detection,
                                           detection_threshold=self.settings.detection_threshold,
                                           save_output=self.settings.save_output, formatter=formatter, resize_with_pad=resize_with_pad,
                                           normalized_detections=normalized_detections, shuffle_indices=shuffle_indices,
                                           squeeze_axis=squeeze_axis, ignore_index=ignore_index, logits_bbox_to_bbox_ls=logits_bbox_to_bbox_ls)
        return transforms

    def get_transform_detection_onnx(self, formatter=None, **kwargs):
        return self.get_transform_detection_base(formatter=formatter, **kwargs)

    def get_transform_detection_mmdet_onnx(self, formatter=None, reshape_list=[(-1,5), (-1,1)],logits_bbox_to_bbox_ls=False, **kwargs):
        return self.get_transform_detection_base(formatter=formatter, reshape_list=reshape_list,logits_bbox_to_bbox_ls=logits_bbox_to_bbox_ls, **kwargs)

    def get_transform_detection_yolov5_onnx(self, formatter=None, **kwargs):
        return self.get_transform_detection_base(formatter=formatter, reshape_list=[(-1,6)], **kwargs)

    def get_transform_detection_yolov5_pose_onnx(self, formatter=None, **kwargs):
        return self.get_transform_detection_base(formatter=formatter, reshape_list=[(-1,57)], **kwargs)

    def get_transform_detection_yolo_6d_object_pose_onnx(self, formatter=None, **kwargs):
        return self.get_transform_detection_base(formatter=formatter, reshape_list=[(-1,15)], **kwargs)

    def get_transform_detection_tv_onnx(self, formatter=DetectionBoxSL2BoxLS(), reshape_list=[(-1,4), (-1,1), (-1,1)],
            squeeze_axis=None, normalized_detections=True, **kwargs):
        return self.get_transform_detection_base(reshape_list=reshape_list, formatter=formatter,
            squeeze_axis=squeeze_axis, normalized_detections=normalized_detections, **kwargs)

    def get_transform_detection_tflite(self, formatter=DetectionYXYX2XYXY(), **kwargs):
        return self.get_transform_detection_base(formatter=formatter, **kwargs)

    def get_transform_detection_mxnet(self, formatter=None, resize_with_pad=False,
                        normalized_detections=False, shuffle_indices=(2,0,1), **kwargs):
        return self.get_transform_detection_base(formatter=formatter, resize_with_pad=resize_with_pad,
                        normalized_detections=normalized_detections, shuffle_indices=shuffle_indices, **kwargs)

    ###############################################################
    # post process transforms for segmentation
    ###############################################################
    def get_transform_segmentation_base(self, data_layout=None, with_argmax=True, **kwargs):
        postprocess_segmentation = [SqueezeAxis()]
        if with_argmax:
            postprocess_segmentation += [ArgMax(axis=None, data_layout=data_layout)]
        #
        postprocess_segmentation += [NPTensorToImage(data_layout=data_layout),
                                     SegmentationImageResize(),
                                     SegmentationImagetoBytes()]
        if self.settings.save_output:
            postprocess_segmentation += [SegmentationImageSave(self.settings.num_output_frames)]
        #
        transforms = PostProcessTransforms(None, postprocess_segmentation,
                                           data_layout=data_layout,
                                           save_output=self.settings.save_output,
                                           with_argmax=with_argmax)
        return transforms

    def get_transform_segmentation_onnx(self, data_layout=constants.NCHW, with_argmax=True):
        return self.get_transform_segmentation_base(data_layout=data_layout, with_argmax=with_argmax)

    def get_transform_segmentation_tflite(self, data_layout=constants.NHWC, with_argmax=True):
        return self.get_transform_segmentation_base(data_layout=data_layout, with_argmax=with_argmax)

    ###############################################################
    # post process transforms for human pose estimation
    ###############################################################
    def get_transform_human_pose_estimation_base(self, data_layout=None, with_udp=True, **kwargs):
        # channel_axis = -1 if data_layout == constants.NHWC else 1
        # postprocess_human_pose_estimation = [SqueezeAxis()] #just removes the first axis from output list, final size (c,w,h)
        postprocess_human_pose_estimation = [HumanPoseHeatmapParser(use_udp=with_udp),
                                             KeypointsProject2Image(use_udp=with_udp)]

        if self.settings.save_output:
            postprocess_human_pose_estimation += [HumanPoseImageSave(self.settings.num_output_frames)]
        #
        transforms = PostProcessTransforms(None, postprocess_human_pose_estimation,
                                           data_layout=data_layout,
                                           save_output=self.settings.save_output)
        return transforms

    def get_transform_human_pose_estimation_onnx(self, data_layout=constants.NCHW):
        return self.get_transform_human_pose_estimation_base(data_layout=data_layout, with_udp=self.settings.with_udp)

    ###############################################################
    # post process transforms for depth estimation
    ###############################################################
    def get_transform_depth_estimation_base(self, data_layout=None, **kwargs):
        postprocess_depth_estimation = [SqueezeAxis(),
                                        NPTensorToImage(data_layout=data_layout),
                                        DepthImageResize()]
        if self.settings.save_output:
            postprocess_depth_estimation += [DepthImageSave(self.settings.num_output_frames)]
        #
        transforms = PostProcessTransforms(None, postprocess_depth_estimation,
                                           data_layout=data_layout,
                                           save_output=self.settings.save_output)
        return transforms

    def get_transform_depth_estimation_onnx(self, data_layout=constants.NCHW, **kwargs):
        return self.get_transform_depth_estimation_base(data_layout=data_layout)

    def get_transform_lidar_base(self, **kwargs):
        postprocess_lidar = [
            OD3DOutPutPorcess(self.settings.detection_threshold)
        ]
        transforms = PostProcessTransforms(None, postprocess_lidar)
        return transforms

    ###############################################################
    # post process transforms for disparity estimation
    ###############################################################
    def get_transform_disparity_estimation_base(self, data_layout):
        postprocess_disparity_estimation = [SqueezeAxis(),
                                            NPTensorToImage(data_layout=data_layout)]
        
        # To REVISIT!
        #if self.settings.save_output:
        #    postprocess_disparity_estimation += [DepthImageSave(self.settings.num_output_frames)]
        #
        transforms = PostProcessTransforms(None, postprocess_disparity_estimation,
                                           data_layout=data_layout,
                                           save_output=self.settings.save_output)
        return transforms

    def get_transform_disparity_estimation_onnx(self, data_layout=constants.NCHW):
        return self.get_transform_disparity_estimation_base(data_layout=data_layout)

