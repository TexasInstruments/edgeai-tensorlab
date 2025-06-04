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
    bev_frame_cfg_ps = {
        'task_type': 'bev_detection',
        'dataset_category': datasets.DATASET_CATEGORY_PANDASET_FRAME,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PANDASET_FRAME]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PANDASET_FRAME]['input_dataset'],
    }

    bev_mv_image_cfg_ps = {
        'task_type': 'bev_detection',
        'dataset_category': datasets.DATASET_CATEGORY_PANDASET_MV_IMAGE,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PANDASET_MV_IMAGE]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PANDASET_MV_IMAGE]['input_dataset'],
    }

    # to define the names of first and last layer for 16 bit conversion
    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        ################# onnx models ###############################
        ## 3dod-8120: PETR for pandaset
        #'3dod-8120':utils.dict_update(bev_frame_cfg_ps,
        #    task_name='PETRv1',
        #    # crop = (left, top, width, height)
        #    preprocess=preproc_transforms.get_transform_bev_petr((1080, 1920), (540, 960), (0, 188, 960, 352), featsize=(22, 60), backend='cv2', interpolation=cv2.INTER_CUBIC),
        #    # Check RGB vs BGR
        #    session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=[(103.530, 116.280, 123.675)], input_scale=[(0.017429, 0.017507, 0.017125)], input_optimization=False,
        #                                                                deny_list_from_start_end_node = {'/pts_bbox_head/Concat_102':None,
        #                                                                                                 '/pts_bbox_head/Concat_101':None,
        #                                                                                                 '/pts_bbox_head/transformer/Transpose_2':'/pts_bbox_head/transformer/Transpose_2',
        #                                                                                                 '/pts_bbox_head/ScatterND_6':'/pts_bbox_head/Unsqueeze_164',
        #                                                                                                 '/pts_bbox_head/ScatterND_12':'/pts_bbox_head/Unsqueeze_165',
        #                                                                                                 '/pts_bbox_head/ScatterND_18':'/pts_bbox_head/Unsqueeze_166',
        #                                                                                                 '/pts_bbox_head/ScatterND_24':'/pts_bbox_head/Unsqueeze_167',
        #                                                                                                 '/pts_bbox_head/ScatterND_30':'/pts_bbox_head/Unsqueeze_168',
        #                                                                                                 '/pts_bbox_head/ScatterND_36':'/pts_bbox_head/Unsqueeze_169'}),
        #        runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
        #            {'advanced_options:output_feature_16bit_names_list':''},
        #            {'advanced_options:max_num_subgraph_nodes':300}),
        #        model_path=f'../edgeai-modelforest/models-cl/vision/detection_3d/pandaset/petr/petrv1_mod_pandaset_vovnet_352x960_20250509.onnx'),
        #    postprocess=postproc_transforms.get_transform_bev_detection_base(),
        #    metric=dict(),
        #    model_info=dict(metric_reference={'mAP':0.4})
        #),
        ## 3dod-8130: BEVDet for Pandaset
        #'3dod-8130':utils.dict_update(bev_frame_cfg_ps,
        #    task_name='BEVDet',
        #    # crop = (left, top, width, height)
        #    preprocess=preproc_transforms.get_transform_bev_bevdet((1080, 1920), (468, 832), (0, 180, 832, 288), backend='cv2', interpolation=cv2.INTER_CUBIC),
        #    session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=[(123.675, 116.280, 103.530)], input_scale=[(0.017125, 0.017507, 0.017429)], input_optimization=False),
        #        runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
        #            {'advanced_options:output_feature_16bit_names_list':''}),
        #        model_path=f'../edgeai-modelforest/models-cl/vision/detection_3d/pandaset/bevdet/bevdet_r50_mod_pandaset_288x832_20250512.onnx'),
        #    postprocess=postproc_transforms.get_transform_bev_detection_bevdet(),
        #    metric=dict(),
        #    model_info=dict(metric_reference={'mAP':0.4})
        #),
        ## 3dod-8140: BEVFormer for pandaset
        #'3dod-8140':utils.dict_update(bev_frame_cfg_ps,
        #    task_name='BEVFormer',
        #    # pad = (left, top, right, bottom) = (0, 0, 0, 30)
        #    preprocess=preproc_transforms.get_transform_bev_bevformer((1080, 1920), (540, 960), (0, 0, 0, 4), backend='cv2', interpolation=cv2.INTER_CUBIC),
        #    session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=[(123.675, 116.280, 103.530)], input_scale=[(0.017125, 0.017507, 0.017429)], input_optimization=False),
        #        runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
        #            {'advanced_options:output_feature_16bit_names_list':''}),
        #        model_path=f'../edgeai-modelforest/models-cl/vision/detection_3d/pandaset/bevformer/bevformer_tiny_mod_pandaset_544x960_20250519.onnx'),
        #    postprocess=postproc_transforms.get_transform_bev_detection_base(),
        #    metric=dict(),
        #    model_info=dict(metric_reference={'mAP':0.4})
        #),
        ## for pandaset the transforms are different
        #'3dod-8150':utils.dict_update(bev_mv_image_cfg_ps,
        #    task_name='FCOS3D',
        #    # pad = (left, top, right, bottom) = (0, 0, 0, 28)
        #    preprocess=preproc_transforms.get_transform_fcos3d((1080,1920), (1080,1920), (0, 0, 0, 8), backend='cv2', interpolation=cv2.INTER_CUBIC),
        #    session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=[(103.530, 116.280, 123.675)], input_scale=[(1.0, 1.0, 1.0)], input_optimization=False),
        #        runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
        #            {'advanced_options:output_feature_16bit_names_list':''}),
        #        model_path=f'../edgeai-modelforest/models-cl/vision/detection_3d/pandaset/fcos3d/fcos3d_mod_pandaset_r101_928x1600_20250509.onnx'),
        #    postprocess=postproc_transforms.get_transform_fcos3d(),
        #    metric=dict(),
        #    model_info=dict(metric_reference={'mAP':0.4})
        #),
        # 3dod-8160: FastBEV without temporal frame for pandaset
        '3dod-8160':utils.dict_update(bev_frame_cfg_ps,
            task_name='FastBEV_f1',
            # crop = (left, top, width, height)
            preprocess=preproc_transforms.get_transform_bev_fastbev((1080,1920), (396,704), (0, 70, 704, 256), backend='cv2', interpolation=cv2.INTER_CUBIC),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=[(123.675, 116.280, 103.530)], 
                                                                      input_scale=[(0.017125, 0.017507, 0.017429)], input_optimization=False,
                                                                      deny_list_from_start_end_node = {'/TopK':None,
                                                                                                       '/Concat_20':'/Concat_20',
                                                                                                       '/Gather_9':'/Gather_9',}),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(bev_options={'bev_options:num_temporal_frames': 0}),
                    {'advanced_options:output_feature_16bit_names_list':'/bbox_head/conv_cls/Conv_output_0, /bbox_head/conv_dir_cls/Conv_output_0, /bbox_head/conv_reg/Conv_output_0'}),
                model_path=f'{settings.models_path}/vision/detection_3d/pandaset/mmdet3d/fastbev/fastbev_mod_pandaset_r18_f1_256x704_20250507.onnx'),
            postprocess=postproc_transforms.get_transform_bev_detection_fastbev(),
            metric=dict(),
            model_info=dict(metric_reference={'mAP':0.4})
        ),

        # 3dod-8161: FastBEV w/ NMS without temporal frame for pandaset
        '3dod-8161':utils.dict_update(bev_frame_cfg_ps,
            task_name='FastBEV_f1',
            # crop = (left, top, width, height)
            preprocess=preproc_transforms.get_transform_bev_fastbev((1080,1920), (396,704), (0, 70, 704, 256), backend='cv2', interpolation=cv2.INTER_CUBIC),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=[(123.675, 116.280, 103.530)], 
                                                                      input_scale=[(0.017125, 0.017507, 0.017429)], input_optimization=False,
                                                                      deny_list_from_start_end_node = {}),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(bev_options={'bev_options:num_temporal_frames': 0},
                    ext_options={'object_detection:meta_arch_type': 7,
                                 'object_detection:meta_layers_names_list':
                                 f'{settings.models_path}/vision/detection_3d/pandaset/mmdet3d/fastbev/fastbev_mod_pandaset_nms_r18_f1_metaarch.prototxt'}),
                    {'advanced_options:output_feature_16bit_names_list':'/bbox_head/conv_cls/Conv_output_0, /bbox_head/conv_dir_cls/Conv_output_0, /bbox_head/conv_reg/Conv_output_0'}),
                model_path=f'{settings.models_path}/vision/detection_3d/pandaset/mmdet3d/fastbev/fastbev_mod_pandaset_nms_r18_f1_256x704_20250507.onnx'),
            postprocess=postproc_transforms.get_transform_bev_detection_fastbev(enable_nms=False),
            metric=dict(),
            model_info=dict(metric_reference={'mAP':0.4})
        ),
    }

    return pipeline_configs
