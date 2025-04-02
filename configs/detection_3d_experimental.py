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

import cv2
from edgeai_benchmark import constants, utils, datasets, preprocess, sessions, postprocess, metrics


def get_configs(settings, work_dir):
    # get the sessions types to use for each model type
    onnx_session_type = settings.get_session_type(constants.MODEL_TYPE_ONNX)

    preproc_transforms = preprocess.PreProcessTransforms(settings)
    postproc_transforms = postprocess.PostProcessTransforms(settings)

    # configs for each model pipeline
    common_cfg_1class = {
        'task_type': 'detection_3d',
        'dataset_category': datasets.DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS]['input_dataset'],
        'postprocess': None
    }

    common_cfg_3class = {
        'task_type': 'detection_3d',
        'dataset_category': datasets.DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS,
        'calibration_dataset': settings.dataset_cache['kitti_lidar_det_3class']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['kitti_lidar_det_3class']['input_dataset'],
        'postprocess': None
    }

    bev_frame_cfg = {
        'task_type': 'bev_detection',
        'dataset_category': datasets.DATASET_CATEGORY_NUSCENES_FRAME,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_NUSCENES_FRAME]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_NUSCENES_FRAME]['input_dataset'],
    }

    bev_mv_image_cfg = {
        'task_type': 'bev_detection',
        'dataset_category': datasets.DATASET_CATEGORY_NUSCENES_MV_IMAGE,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_NUSCENES_MV_IMAGE]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_NUSCENES_MV_IMAGE]['input_dataset'],
    }

    # to define the names of first and last layer for 16 bit conversion
    first_last_layer_3dod_7100 = ''

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        ################# onnx models ###############################
        '3dod-7100':utils.dict_update(common_cfg_1class,
            preprocess=preproc_transforms.get_transform_lidar_base(),
            session=onnx_session_type(**sessions.get_nomeanscale_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_qat_v1(det_options=True,
                    ext_options={'object_detection:meta_arch_type': 7,
                     'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection_3d/kitti/mmdet3d/lidar_point_pillars_10k_496x432.prototxt',
                     'advanced_options:output_feature_16bit_names_list': first_last_layer_3dod_7100}),
                model_path=f'{settings.models_path}/vision/detection_3d/kitti/mmdet3d/lidar_point_pillars_10k_496x432_qat-p2.onnx'),
            postprocess=postproc_transforms.get_transform_lidar_base(),
            metric=dict(label_offset_pred=None),
            model_info=dict(metric_reference={'accuracy_ap_3d_moderate%':76.50}, model_shortlist=200, compact_name='pointPillars-lidar-10000-1c-496x432', shortlisted=True)
        ),
        '3dod-7110':utils.dict_update(common_cfg_3class,
            preprocess=preproc_transforms.get_transform_lidar_base(),
            session=onnx_session_type(**sessions.get_nomeanscale_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_qat_v1(det_options=True,
                    ext_options={'object_detection:meta_arch_type': 7,
                     'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection_3d/kitti/mmdet3d/lidar_point_pillars_10k_496x432_3class.prototxt',
                     'advanced_options:output_feature_16bit_names_list': first_last_layer_3dod_7100}),
                model_path=f'{settings.models_path}/vision/detection_3d/kitti/mmdet3d/lidar_point_pillars_10k_496x432_3class_qat-p2.onnx'),
            postprocess=postproc_transforms.get_transform_lidar_base(),
            metric=dict(label_offset_pred=None),
            model_info=dict(metric_reference={'accuracy_ap_3d_moderate%':76.50}, model_shortlist=None, compact_name='pointPillars-lidar-10000-3c-496x432', shortlisted=False)
        ),
        # 3dod-7120: PETR
        '3dod-7120':utils.dict_update(bev_frame_cfg,
            task_name='PETRv1',
            # To double check image param changes due to scaling and cropping
            preprocess=preproc_transforms.get_transform_bev_petr((900, 1600), (450, 800), (0, 130, 800, 320), backend='cv2', interpolation=cv2.INTER_CUBIC),
            # Check RGB vs BGR
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(103.530, 116.280, 123.675), input_scale=(0.017429, 0.017507, 0.017125),
                                                                        input_optimization=False,
                                                                        deny_list_from_start_end_node = {'/pts_bbox_head/Concat_102':None,
                                                                                                         '/pts_bbox_head/Concat_101':None,
                                                                                                         '/pts_bbox_head/transformer/Transpose_2':'/pts_bbox_head/transformer/Transpose_2',
                                                                                                         '/pts_bbox_head/ScatterND_6':'/pts_bbox_head/Unsqueeze_164',
                                                                                                         '/pts_bbox_head/ScatterND_12':'/pts_bbox_head/Unsqueeze_165',
                                                                                                         '/pts_bbox_head/ScatterND_18':'/pts_bbox_head/Unsqueeze_166',
                                                                                                         '/pts_bbox_head/ScatterND_24':'/pts_bbox_head/Unsqueeze_167',
                                                                                                         '/pts_bbox_head/ScatterND_30':'/pts_bbox_head/Unsqueeze_168',
                                                                                                         '/pts_bbox_head/ScatterND_36':'/pts_bbox_head/Unsqueeze_169'}),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
                    {'advanced_options:output_feature_16bit_names_list':''},
                    {'advanced_options:max_num_subgraph_nodes':300}),
                model_path=f'../edgeai-modelforest/models-cl/vision/detection_3d/nuscenes/petr/edgeai_petrv1_vovnet_p4_320x800.onnx'),
            postprocess=postproc_transforms.get_transform_bev_detection_base(),
            metric=dict(),
            model_info=dict(metric_reference={'mAP':0.4})
        ),
        # 3dod-7130: BEVDet
        '3dod-7130':utils.dict_update(bev_frame_cfg,
            task_name='BEVDet',
            # crop = (left, top, width, height)
            preprocess=preproc_transforms.get_transform_bev_bevdet((900, 1600), (396, 704), (0, 140, 704, 256), backend='cv2', interpolation=cv2.INTER_CUBIC),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(123.675, 116.280, 103.530), input_scale=(0.017125, 0.017507, 0.017429)),
                input_optimization=False,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
                    {'advanced_options:output_feature_16bit_names_list':''}),
                model_path=f'../edgeai-modelforest/models-cl/vision/detection_3d/nuscenes/bevdet/edgeai_bevdet_tiny_res50_256x704.onnx'),
            postprocess=postproc_transforms.get_transform_bev_detection_bevdet(),
            metric=dict(),
            model_info=dict(metric_reference={'mAP':0.4})
        ),
        # 3dod-7140: BEVFormer
        '3dod-7140':utils.dict_update(bev_frame_cfg,
            task_name='BEVFormer',
            # pad = (left, top, right, bottom) = (0, 0, 0, 30)
            preprocess=preproc_transforms.get_transform_bev_bevformer((900, 1600), (450, 800), (0, 0, 0, 30), backend='cv2', interpolation=cv2.INTER_CUBIC),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(123.675, 116.280, 103.530), input_scale=(0.017125, 0.017507, 0.017429)),
                input_optimization=False,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
                    {'advanced_options:output_feature_16bit_names_list':''}),
                model_path=f'../edgeai-modelforest/models-cl/vision/detection_3d/nuscenes/bevformer/edgeai_bevformer_tiny_480x800.onnx'),
            postprocess=postproc_transforms.get_transform_bev_detection_base(),
            metric=dict(),
            model_info=dict(metric_reference={'mAP':0.4})
        ),
        # 3dod-7150: FCOS3D (bev_mv_image_cfg)
        '3dod-7150':utils.dict_update(bev_mv_image_cfg,
            task_name='FCOS3D',
            # pad = (left, top, right, bottom) = (0, 0, 0, 28)
            preprocess=preproc_transforms.get_transform_fcos3d((900, 1600), (900, 1600), (0, 0, 0, 28), backend='cv2', interpolation=cv2.INTER_CUBIC),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(103.530, 116.280, 123.675), input_scale=(1.0, 1.0, 1.0)),
                input_optimization=False,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
                    {'advanced_options:output_feature_16bit_names_list':''}),
                model_path=f'../edgeai-modelforest/models-cl/vision/detection_3d/nuscenes/fcos3d/fcos3d_r101_928x1600.onnx'),
            postprocess=postproc_transforms.get_transform_fcos3d(),
            metric=dict(),
            model_info=dict(metric_reference={'mAP':0.4})
        ),
        # 3dod-7160: FastBEV with 1 temporal frame
        '3dod-7160':utils.dict_update(bev_frame_cfg,
            task_name='FastBEV_f1',
            # crop = (left, top, width, height)
            preprocess=preproc_transforms.get_transform_bev_fastbev((900, 1600), (396, 704), (0, 70, 704, 256), backend='cv2', interpolation=cv2.INTER_CUBIC),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(123.675, 116.280, 103.530), input_scale=(0.017125, 0.017507, 0.017429),
                                                                        input_optimization=False,
                                                                        deny_list_from_start_end_node = {'/TopK':None,
                                                                                                         '/Concat_20':'/Concat_20', 
                                                                                                         '/Gather_9':'/Gather_9'}),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(bev_options={'bev_options:num_temporal_frames': 0}),
                    {'advanced_options:output_feature_16bit_names_list':''}),
                model_path=f'../edgeai-modelforest/models-cl/vision/detection_3d/nuscenes/fastbev/edgeai_fastbev_r18_c192_d2_f1_256x704.onnx'),
            postprocess=postproc_transforms.get_transform_bev_detection_fastbev(),
            metric=dict(),
            model_info=dict(metric_reference={'mAP':0.4})
        ),
        # 3dod-7161: FastBEV with 4 temporal frames
        '3dod-7161':utils.dict_update(bev_frame_cfg,
            task_name='FastBEV_f4',
            # crop = (left, top, width, height)
            preprocess=preproc_transforms.get_transform_bev_fastbev((900, 1600), (396, 704), (0, 70, 704, 256), backend='cv2', interpolation=cv2.INTER_CUBIC),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(123.675, 116.280, 103.530), input_scale=(0.017125, 0.017507, 0.017429),
                                                                        input_optimization=False,
                                                                        deny_list_from_start_end_node = {'/TopK':None,
                                                                                                         '/Concat_7':'Concat_7',
                                                                                                         '/Gather_2':'/Gather_2',
                                                                                                         '/Gather_6':'/Gather_6',
                                                                                                         '/Gather_10':'/Gather_10',
                                                                                                         '/Gather_14':'/Gather_14',
                                                                                                         '/Slice_3':'/Slice_3',
                                                                                                         '/Slice_8':'/Slice_8',
                                                                                                         '/Concat_30':'/Concat_30',
                                                                                                         '/Gather_22':'/Gather_22',
                                                                                                         }),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(bev_options={'bev_options:num_temporal_frames': 3}),
                    {'advanced_options:output_feature_16bit_names_list':''}),
                model_path=f'../edgeai-modelforest/models-cl/vision/detection_3d/nuscenes/fastbev/edgeai_fastbev_r34_c224_d4_f4_256x704.onnx'),
            postprocess=postproc_transforms.get_transform_bev_detection_fastbev(),
            metric=dict(),
            model_info=dict(metric_reference={'mAP':0.4})
        )
    }

    return pipeline_configs
