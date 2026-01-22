from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn

import copy
import onnx
from onnxsim import simplify

from mmengine.model.base_module import BaseModule
from mmdet3d.registry import MODELS


# To export each head separately
# Recommend to diable save_onnx_model, and
# manually set the followings to True:
#   SparseDriveHead.det_head_export
#   SparseDriveHead.map_head_export
#   SparseDriveHead.motion_plan_head_export
def export_det_head(det_head_onnx, feature_maps, metas, time_interval, T_temp2cur):
    model_input = []
    model_input.append(feature_maps)
    model_input.append(metas['projection_mat'].to(feature_maps[0].device))
    model_input.append(metas['gt_ego_fut_cmd'].to(feature_maps[0].device))
    for _, val in det_head_onnx.det_history.items():
        model_input.append(val)
    model_input.append(time_interval)
    model_input.append(T_temp2cur)

    input_names=["img_feats_0", "img_feats_1", "img_feats_2", "img_feats_3",
                 "projection_mat", "gt_ego_fut_cmd",
                 "det_cached_feature", "det_cached_anchor", "det_prev_id", "det_instance_id", "det_confidence", "det_temp_confidence",
                 "time_interval", "T_temp2cur"]
    output_names=["classification", "prediction", "quality", "instance_feature", "anchor_embed", "instance_id",
                  "det_his_cached_feature", "det_his_cached_anchor", "det_his_prev_id", "det_his_instance_id", "det_his_confidence", "det_his_temp_confidence"]

    model_name = "det_head.onnx"
    torch.onnx.export(
        det_head_onnx,
        tuple(model_input),
        model_name,
        opset_version=16,
        input_names=input_names,
        output_names=output_names)
    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    print("\n{} is exported".format(model_name))


class DetHeadONNX(nn.Module):
    def __init__(self, det_head_module, metas):
        super(DetHeadONNX, self).__init__()
        self.det_head_module = copy.deepcopy(det_head_module).cpu()
        self.metas = metas

        self.det_num_anchor         = det_head_module.instance_bank.num_anchor
        self.det_embed_dims         = det_head_module.instance_bank.embed_dims
        self.det_num_temp_instances = det_head_module.instance_bank.num_temp_instances
        self.num_anchor_feats       = 11

        # timestamp and T_global of the previous metas are not available during export.
        # So use them of the current metas
        self.his_timestamp = self.metas['timestamp']
        self.his_T_global = self.metas['img_metas'][0]['T_global']

        self.det_history = None

    def get_history(self, img_feat):
        if self.det_history is None:
            bs = 1
            device = img_feat.device

            self.det_history = {
                'cached_feature': torch.zeros(bs, self.det_num_temp_instances, self.det_embed_dims).to(device),
                'cached_anchor': torch.zeros(bs, self.det_num_temp_instances, self.num_anchor_feats).to(device),
                'prev_id': torch.tensor(0).int().to(device),
                'instance_id': -1 * torch.zeros(bs, self.det_num_anchor).int().to(device),
                'confidence': torch.zeros(bs, self.det_num_temp_instances).to(device),
                'temp_confidence': torch.zeros(bs, self.det_num_anchor).to(device),
            }

        history_time = self.his_timestamp
        time_interval = self.metas["timestamp"].to(device) - history_time.to(device)
        time_interval = time_interval.to(dtype=torch.float32)
        T_temp2cur = img_feat.new_tensor(
            np.stack(
                [
                    x["T_global_inv"]
                    @ self.his_T_global
                    for i, x in enumerate(self.metas["img_metas"])
                ]
            )
        )

        return time_interval, T_temp2cur


    def forward(self,
                feature_maps=None,
                projection_mat=None,
                gt_ego_fut_cmd=None,
                det_cached_feature=None,
                det_cached_anchor=None,
                det_prev_id=None,
                det_instance_id=None,
                det_confidence=None,
                det_temp_confidence=None,
                time_interval=None,
                T_temp2cur=None):

        metas = {
            'return_loss': False,
            'rescale': True,
            'img_metas': self.metas['img_metas'],
            'projection_mat': projection_mat,
            'image_wh': self.metas['image_wh'].to(feature_maps[0].device),
            'gt_ego_fut_cmd': gt_ego_fut_cmd.to(feature_maps[0].device),
        }

        det_history = {
            'cached_feature': det_cached_feature,
            'cached_anchor': det_cached_anchor,
            'prev_id': det_prev_id,
            'instance_id': det_instance_id,
            'confidence': det_confidence,
            'temp_confidence': det_temp_confidence,
        }

        det_output, det_history_out = self.det_head_module(feature_maps,
                    metas, bank_history=det_history, time_interval=time_interval, T_temp2cur=T_temp2cur)

        det_output['classification'] = torch.cat(det_output['classification'])
        det_output['prediction'] = torch.cat(det_output['prediction'])
        det_output['quality'] = torch.cat(det_output['quality'])

        return det_output, det_history_out


def export_map_head(map_head_onnx, feature_maps, metas, time_interval, T_temp2cur):
    model_input = []
    model_input.append(feature_maps)
    model_input.append(metas['projection_mat'].to(feature_maps[0].device))
    model_input.append(metas['gt_ego_fut_cmd'].to(feature_maps[0].device))
    for _, val in map_head_onnx.map_history.items():
        model_input.append(val)
    model_input.append(time_interval)
    model_input.append(T_temp2cur)

    input_names=["img_feats_0", "img_feats_1", "img_feats_2", "img_feats_3",
                 "projection_mat", "gt_ego_fut_cmd",
                 "map_cached_feature", "map_cached_anchor", "map_prev_id", "map_instance_id", "map_confidence", "map_temp_confidence",
                 "time_interval", "T_temp2cur"]
    # instance_id doesn't exist in map head
    # quality is None. Need to check if quality is always None and how it is handled in a whole model.
    output_names=["classification", "prediction", "instance_feature", "anchor_embed",
                  "map_his_cached_feature", "map_his_cached_anchor", "map_his_prev_id", "map_his_instance_id", "map_his_confidence", "map_his_temp_confidence"]

    model_name = "map_head.onnx"
    torch.onnx.export(
        map_head_onnx,
        tuple(model_input),
        model_name,
        opset_version=16,
        input_names=input_names,
        output_names=output_names)
    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    print("\n{} is exported".format(model_name))


class MapHeadONNX(nn.Module):
    def __init__(self, map_head_module, metas):
        super(MapHeadONNX, self).__init__()
        self.map_head_module = copy.deepcopy(map_head_module).cpu()
        self.metas = metas

        self.map_num_anchor         = map_head_module.instance_bank.num_anchor
        self.map_embed_dims         = map_head_module.instance_bank.embed_dims
        self.map_num_temp_instances = map_head_module.instance_bank.num_temp_instances
        self.num_anchor_feats       = 11

        # timestamp and T_global of the previous metas are not available during export.
        # So use them of the current metas
        self.his_timestamp = self.metas['timestamp']
        self.his_T_global = self.metas['img_metas'][0]['T_global']

        self.map_history = None

    def get_history(self, img_feat):
        if self.map_history is None:
            bs = 1
            device = img_feat.device

            self.map_history = {
                'cached_feature': torch.zeros(bs, self.map_num_temp_instances, self.map_embed_dims).to(device),
                'cached_anchor': torch.zeros(bs, self.map_num_temp_instances, 40).to(device),
                'prev_id': torch.tensor(0).int().to(device),
                'instance_id': -1 * torch.zeros(bs, self.map_num_anchor).int().to(device),
                'confidence': torch.zeros(bs, self.map_num_temp_instances).to(device),
                'temp_confidence': torch.zeros(bs, self.map_num_anchor).to(device),
            }

        history_time = self.his_timestamp
        # timestamp of the previous metas is not available during export.
        # So use timestamp of the current metas
        time_interval = self.metas["timestamp"].to(device) - history_time.to(device)
        time_interval = time_interval.to(dtype=torch.float32)
        T_temp2cur = img_feat.new_tensor(
            np.stack(
                [
                    x["T_global_inv"]
                    @ self.his_T_global
                    for i, x in enumerate(self.metas["img_metas"])
                ]
            )
        )

        return time_interval, T_temp2cur


    def forward(self,
                feature_maps=None,
                projection_mat=None,
                gt_ego_fut_cmd=None,
                map_cached_feature=None,
                map_cached_anchor=None,
                map_prev_id=None,
                map_instance_id=None,
                map_confidence=None,
                map_temp_confidence=None,
                time_interval=None,
                T_temp2cur=None):

        metas = {
            'return_loss': False,
            'rescale': True,
            'img_metas': self.metas['img_metas'],
            'projection_mat': projection_mat,
            'image_wh': self.metas['image_wh'].to(feature_maps[0].device),
            'gt_ego_fut_cmd': gt_ego_fut_cmd.to(feature_maps[0].device),
        }

        map_history = {
            'cached_feature': map_cached_feature,
            'cached_anchor': map_cached_anchor,
            'prev_id': map_prev_id,
            'instance_id': map_instance_id,
            'confidence': map_confidence,
            'temp_confidence': map_temp_confidence,
        }

        map_output, map_history_out = self.map_head_module(feature_maps,
                    metas, bank_history=map_history, time_interval=time_interval, T_temp2cur=T_temp2cur)

        map_output['classification'] = torch.cat(map_output['classification'])
        map_output['prediction'] = torch.cat(map_output['prediction'])
        if all(item is not None for item in map_output['quality']):
            map_output['quality'] = torch.cat(map_output['quality'])

        return map_output, map_history_out


def export_motion_plan_head(motion_plan_head_onnx,
                            det_output, map_output, feature_maps, metas,
                            time_interval, T_temp2cur):
    model_input = []
    model_input.append(det_output)
    model_input.append(map_output)
    model_input.append(feature_maps)
    model_input.append(metas['projection_mat'].to(feature_maps[0].device))
    model_input.append(metas['gt_ego_fut_cmd'].to(feature_maps[0].device))
    for _, val in motion_plan_head_onnx.motion_history.items():
        model_input.append(val)
    model_input.append(time_interval)
    model_input.append(T_temp2cur)

    input_names=["det_classification_0", "det_classification_1", "det_classification_2",  "det_classification_3",  "det_classification_4", "det_classification_5",
                 "det_prediction_0", "det_prediction_1", "det_prediction_2", "det_prediction_3", "det_prediction_4", "det_prediction_5",
                 "det_quality_0", "det_quality_1", "det_quality_2", "det_quality_3", "det_quality_4", "det_quality_5", 
                 "det_instance_feature", "det_anchor_embed", "det_instance_id",
                 "map_classification_0", "map_classification_1", "map_classification_2",  "map_classification_3",  "map_classification_4", "map_classification_5",
                 "map_prediction_0", "map_prediction_1", "map_prediction_2", "map_prediction_3", "map_prediction_4", "map_prediction_5",
                 #"map_quality_0", "map_quality_1", "map_quality_2", "map_quality_3", "map_quality_4", "map_quality_5",
                 "map_instance_feature", "map_anchor_embed",
                 "img_feats_0", "img_feats_1", "img_feats_2", "img_feats_3",
                 "projection_mat", "gt_ego_fut_cmd",
                 "motion_prev_instance_id", "motion_prev_confidence", "motion_period_in", "motion_instance_feature_queue", "motion_anchor_queue_in",
                 "motion_prev_ego_status", "motion_ego_period", "motion_ego_feature_queue", "motion_ego_anchor_queue",
                 "time_interval", "T_temp2cur"]
    output_names=["motion_classification", "motion_prediction", "motion_period", "motion_anchor_queue",
                  "planning_classification", "planning_prediction", "planning_status", "planning_period", "planning_anchor_queue",
                  "motion_his_prev_instance_id", "motion_his_prev_confidence", "motion_his_period", "motion_his_instance_feature_queue", "motion_his_anchor_queue",
                  "motion_his_prev_ego_status", "motion_his_ego_period", "motion_his_ego_feature_queue", "motion_his_ego_anchor_queue"]

    model_name = "motion_plan_head.onnx"
    torch.onnx.export(
        motion_plan_head_onnx,
        tuple(model_input),
        model_name,
        opset_version=16,
        input_names=input_names,
        output_names=output_names
        )
    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    print("\n{} is exported".format(model_name))


class MotionPlanHeadONNX(nn.Module):
    def __init__(self, motion_plan_head_module, metas, num_anchor, anchor_encoder, mask, anchor_handler):
        super(MotionPlanHeadONNX, self).__init__()
        self.motion_plan_head_module = copy.deepcopy(motion_plan_head_module).cpu()
        self.metas = metas
        self.anchor_encoder = copy.deepcopy(anchor_encoder)
        self.mask = torch.Tensor([True]).to(bool) # set fixed value to export model #mask 
        self.anchor_handler = copy.deepcopy(anchor_handler)

        self.num_anchor             = num_anchor
        self.embed_dims             = motion_plan_head_module.instance_queue.embed_dims
        self.queue_length           = motion_plan_head_module.instance_queue.queue_length

        self.num_anchor_feats       = 11
        self.num_ego_feats          = 10

        # timestamp and T_global of the previous metas are not available during export.
        # So use them of the current metas
        self.his_timestamp = self.metas['timestamp']
        self.his_T_global = self.metas['img_metas'][0]['T_global']

        self.motion_history = None

    def get_history(self, img_feat):
        if self.motion_history is None:
            bs = 1
            device = img_feat.device

            self.motion_history = {
                'prev_instance_id': -1 * torch.zeros(bs, self.num_anchor).int().to(device),
                'prev_confidence': torch.zeros(bs, self.num_anchor).to(device),
                'period': torch.zeros(bs, self.num_anchor).to(device),
                'instance_feature_queue': torch.zeros(1, bs, self.num_anchor, self.embed_dims).to(device),
                'anchor_queue': torch.zeros(1, bs, self.num_anchor, self.num_anchor_feats).to(device),
                'prev_ego_status': torch.zeros(bs, 1, self.num_ego_feats).to(device),
                'ego_period': torch.tensor(0).int().to(device),
                'ego_feature_queue': torch.zeros(1, bs, 1, self.embed_dims).to(device),
                'ego_anchor_queue': torch.zeros(1, bs, 1, self.num_anchor_feats).to(device),
            }

        history_time = self.his_timestamp
        # timestamp of the previous metas is not available during export.
        # So use timestamp of the current metas
        time_interval = self.metas["timestamp"].to(device) - history_time.to(device)
        time_interval = time_interval.to(dtype=torch.float32)
        T_temp2cur = img_feat.new_tensor(
            np.stack(
                [
                    x["T_global_inv"]
                    @ self.his_T_global
                    for i, x in enumerate(self.metas["img_metas"])
                ]
            )
        )

        return time_interval, T_temp2cur


    def forward(self,
                det_output=None,
                map_output=None,
                feature_maps=None,
                projection_mat=None,
                gt_ego_fut_cmd=None,
                motion_prev_instance_id=None,
                motion_prev_confidence=None,
                motion_period=None,
                motion_instance_feature_queue=None,
                motion_anchor_queue=None,
                motion_prev_ego_status=None,
                motion_ego_period=None,
                motion_ego_feature_queue=None,
                motion_ego_anchor_queue=None,
                time_interval=None,
                T_temp2cur=None):

        metas = {
            'return_loss': False,
            'rescale': True,
            'img_metas': self.metas['img_metas'],
            'projection_mat': projection_mat,
            'image_wh': self.metas['image_wh'].to(feature_maps[0].device),
            'gt_ego_fut_cmd': gt_ego_fut_cmd.to(feature_maps[0].device),
        }

        motion_history = {
            'prev_instance_id': motion_prev_instance_id,
            'prev_confidence': motion_prev_confidence,
            'period': motion_period,
            'instance_feature_queue': motion_instance_feature_queue,
            'anchor_queue': motion_anchor_queue,
            'prev_ego_status': motion_prev_ego_status,
            'ego_period': motion_ego_period,
            'ego_feature_queue': motion_ego_feature_queue,
            'ego_anchor_queue': motion_ego_anchor_queue,
        }

        motion_output, planning_output, motion_history_out = \
            self.motion_plan_head_module(
                det_output,
                map_output,
                feature_maps,
                metas,
                self.anchor_encoder.to(feature_maps[0].device),
                self.mask.to(feature_maps[0].device),
                self.anchor_handler.to(feature_maps[0].device),
                queue_history=motion_history,
                time_interval=time_interval,
                T_temp2cur=T_temp2cur)

        motion_output['anchor_queue'] = torch.cat(motion_output['anchor_queue'])
        planning_output['anchor_queue'] = torch.cat(planning_output['anchor_queue'])

        return motion_output, planning_output, motion_history_out


@MODELS.register_module()
class SparseDriveHead(BaseModule):
    def __init__(
        self,
        task_config: dict,
        det_head = dict,
        map_head = dict,
        motion_plan_head = dict,
        init_cfg=None,
        **kwargs,
    ):
        super(SparseDriveHead, self).__init__(init_cfg)
        self.task_config = task_config
        if self.task_config['with_det']:
            self.det_head = MODELS.build(det_head)
        if self.task_config['with_map']:
            self.map_head = MODELS.build(map_head)
        if self.task_config['with_motion_plan']:
            self.motion_plan_head = MODELS.build(motion_plan_head)
        self.det_head_export = False
        self.map_head_export = False
        self.motion_plan_head_export = False

    def init_weights(self):
        if self.task_config['with_det']:
            self.det_head.init_weights()
        if self.task_config['with_map']:
            self.map_head.init_weights()
        if self.task_config['with_motion_plan']:
            self.motion_plan_head.init_weights()

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
        det_history=None,
        map_history=None,
        motion_history=None,
        time_interval=None,
        T_temp2cur=None
    ):
        det_history_out = None
        map_history_out = None
        motion_history_out = None

        if self.task_config['with_det']:
            if torch.onnx.is_in_onnx_export():
                det_output, det_history_out = self.det_head(
                    feature_maps, metas,
                    bank_history=det_history, time_interval=time_interval, T_temp2cur=T_temp2cur)
            else:
                # Export det_head
                if self.det_head_export is True:
                    det_head_module = self.det_head
                    det_head_onnx = DetHeadONNX(det_head_module, metas)
                    det_head_onnx.eval()

                    # to cpu
                    for i in range(len(feature_maps)):
                        feature_maps[i] = feature_maps[i].cpu()
                    # export
                    time_interval, T_temp2cur = det_head_onnx.get_history(feature_maps[0])
                    export_det_head(det_head_onnx, feature_maps, metas, time_interval, T_temp2cur)

                    # to gpu
                    for i in range(len(feature_maps)):
                        feature_maps[i] = feature_maps[i].cuda()
                    self.det_head_export = False

                det_output = self.det_head(feature_maps, metas)
        else:
            det_output = None

        if self.task_config['with_map']:
            if torch.onnx.is_in_onnx_export():
                map_output, map_history_out = self.map_head(feature_maps, metas,
                    bank_history=map_history, time_interval=time_interval, T_temp2cur=T_temp2cur)
            else:
                # Export map_head
                if self.map_head_export is True:
                    map_head_module = self.map_head
                    map_head_onnx = MapHeadONNX(map_head_module, metas)
                    map_head_onnx.eval()

                    # to cpu
                    for i in range(len(feature_maps)):
                        feature_maps[i] = feature_maps[i].cpu()
                    # export
                    time_interval, T_temp2cur = map_head_onnx.get_history(feature_maps[0])
                    export_map_head(map_head_onnx, feature_maps, metas, time_interval, T_temp2cur)

                    # to gpu
                    for i in range(len(feature_maps)):
                        feature_maps[i] = feature_maps[i].cuda()
                    self.map_head_export = False

                map_output = self.map_head(feature_maps, metas)
        else:
            map_output = None

        if self.task_config['with_motion_plan']:
            if torch.onnx.is_in_onnx_export():
                motion_output, planning_output, motion_history_out = self.motion_plan_head(
                    det_output,
                    map_output,
                    feature_maps,
                    metas,
                    self.det_head.anchor_encoder,
                    self.det_head.instance_bank.mask,
                    self.det_head.instance_bank.anchor_handler,
                    queue_history=motion_history,
                    time_interval=time_interval,
                    T_temp2cur=T_temp2cur
                )
            else:
                # Export motion and planning head
                if self.motion_plan_head_export is True:
                    motion_plan_head_module = self.motion_plan_head
                    motion_plan_head_onnx = MotionPlanHeadONNX(
                        motion_plan_head_module,
                        metas,
                        self.det_head.instance_bank.num_anchor,
                        self.det_head.anchor_encoder,
                        self.det_head.instance_bank.mask,
                        self.det_head.instance_bank.anchor_handler)
                    motion_plan_head_onnx.eval()

                    # to cpu
                    for i in range(len(feature_maps)):
                        feature_maps[i] = feature_maps[i].cpu()
                    for key, val in det_output.items():
                        if isinstance(val, list):
                            for i in range(len(val)):
                                if val[i] is not None:
                                    det_output[key][i] = val[i].cpu()
                        else:
                            if val is not None:
                                det_output[key] = val.cpu()
                    for key, val in map_output.items():
                        if isinstance(val, list):
                            for i in range(len(val)):
                                if val[i] is not None:
                                    map_output[key][i] = val[i].cpu()
                        else:
                            if val is not None:
                                map_output[key] = val.cpu()
                    time_interval, T_temp2cur = motion_plan_head_onnx.get_history(feature_maps[0])
                    export_motion_plan_head(motion_plan_head_onnx,
                                            det_output, map_output, feature_maps, metas,
                                            time_interval, T_temp2cur)

                    # to gpu
                    for i in range(len(feature_maps)):
                        feature_maps[i] = feature_maps[i].cuda()
                    for key, val in det_output.items():
                        if isinstance(val, list):
                            for i in range(len(val)):
                                if val[i] is not None:
                                    det_output[key][i] = val[i].cuda()
                        else:
                            if val is not None:
                                det_output[key] = val.cuda()
                    for key, val in map_output.items():
                        if isinstance(val, list):
                            for i in range(len(val)):
                                if val[i] is not None:
                                    map_output[key][i] = val[i].cuda()
                        else:
                            if val is not None:
                                map_output[key] = val.cuda()
                    self.motion_plan_head_export = False

                motion_output, planning_output = self.motion_plan_head(
                    det_output,
                    map_output,
                    feature_maps,
                    metas,
                    self.det_head.anchor_encoder,
                    self.det_head.instance_bank.mask,
                    self.det_head.instance_bank.anchor_handler
                )
        else:
            motion_output, planning_output = None, None

        if torch.onnx.is_in_onnx_export():
            return det_output, map_output, motion_output, planning_output, \
                det_history_out, map_history_out, motion_history_out
        else:
            return det_output, map_output, motion_output, planning_output

    def loss(self, model_outs, data):
        det_output, map_output, motion_output, planning_output = model_outs
        losses = dict()
        if self.task_config['with_det']:
            loss_det = self.det_head.loss(det_output, data)
            losses.update(loss_det)
        
        if self.task_config['with_map']:
            loss_map = self.map_head.loss(map_output, data)
            losses.update(loss_map)

        if self.task_config['with_motion_plan']:
            motion_loss_cache = dict(
                indices=self.det_head.sampler.indices,
            )
            loss_motion = self.motion_plan_head.loss(
                motion_output, 
                planning_output, 
                data, 
                motion_loss_cache
            )
            losses.update(loss_motion)
        
        return losses

    def post_process(self, model_outs, metas):
        det_output, map_output, motion_output, planning_output = model_outs
        if self.task_config['with_det']:
            det_result = self.det_head.post_process(det_output)
            batch_size = len(det_result)
            # Convert bboxes to Lidar Box
            num_samples = len(det_result)
            for i in range(num_samples):
                preds = det_result[i]
                bboxes = preds['bboxes_3d']
                code_size = bboxes.shape[-1]
                bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                if not torch.onnx.is_in_onnx_export():
                    preds['bboxes_3d'] = metas["img_metas"][i]['box_type_3d'](bboxes, code_size)


        if self.task_config['with_map']:
            map_result= self.map_head.post_process(map_output)
            batch_size = len(map_result)

        if self.task_config['with_motion_plan']:
            motion_result, planning_result = self.motion_plan_head.post_process(
                det_output,
                motion_output,
                planning_output,
                metas,
            )

        results = [dict()] * batch_size
        for i in range(batch_size):
            if self.task_config['with_det']:
                results[i].update(det_result[i])
            if self.task_config['with_map']:
                results[i].update(map_result[i])
            if self.task_config['with_motion_plan']:
                results[i].update(motion_result[i])
                results[i].update(planning_result[i])

        return results
