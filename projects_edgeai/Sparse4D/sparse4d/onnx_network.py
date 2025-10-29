import copy
import torch
import torch.nn as nn
import numpy as np

import torchvision.transforms.functional as tF
import torchvision.transforms._functional_tensor as tF_t

#from mmengine.utils import digit_version
#from mmengine.utils.dl_utils import TORCH_VERSION
from .detection3d.box3d import *
from .ops import feature_maps_format


class Sparse4D_export_model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.img_backbone  = model.img_backbone
        self.img_neck      = model.img_neck
        self.depth_branch  = model.depth_branch
        self.pts_bbox_head = model.pts_bbox_head

        self.return_detph  = False
        self.use_deformable_func = model.use_deformable_func
        assert self.return_detph is False, "ONNX export does not support depth prediction."
        assert self.use_deformable_func is False, "ONNX export does not support deformable function."

        self.det_histroy = None
        self.det_num_anchor         = model.pts_bbox_head.instance_bank.num_anchor
        self.det_embed_dims         = model.pts_bbox_head.instance_bank.embed_dims
        self.det_num_temp_instances = model.pts_bbox_head.instance_bank.num_temp_instances
        self.anchor_feats_dims  = VZ + 1

        self.his_timestamp    = None
        self.his_T_global     = None
        self.his_T_global_inv = None

    def prepare_data(self, img_metas):
        self.img_metas = img_metas
        if self.his_timestamp is None:
            self.his_timestamp = self.img_metas[0]['timestamp']
            self.his_T_global = self.img_metas[0]['T_global']
            self.his_T_global_inv = self.img_metas[0]['T_global_inv']

    def get_history(self, img):
        if self.det_histroy is None:
            bs = 1
            device = img.device

            self.det_histroy = {
                'cached_feature': torch.zeros(bs, self.det_num_temp_instances, self.det_embed_dims).to(device),
                'cached_anchor': torch.zeros(bs, self.det_num_temp_instances, self.anchor_feats_dims).to(device),
                'prev_id': torch.tensor(0).int().to(device),
                'instance_id': -1 * torch.zeros(bs, self.det_num_anchor).int().to(device),
                'confidence': torch.zeros(bs, self.det_num_temp_instances).to(device),
                'temp_confidence': torch.zeros(bs, self.det_num_anchor).to(device),
            }

        history_time  = self.his_timestamp
        time_interval = self.img_metas[0]["timestamp"] - history_time
        T_temp2cur = img.new_tensor(
            np.stack(
                [
                    x["T_global_inv"]
                    @ self.his_T_global
                    for i, x in enumerate(self.img_metas)
                ]
            )
        )

        return time_interval, T_temp2cur

    def forward(self,
                img,
                projection_mat=None,
                det_cached_feature=None,
                det_cached_anchor=None,
                det_prev_id=None,
                det_instance_id=None,
                det_confidence=None,
                det_temp_confidence=None,
                time_interval=None,
                T_temp2cur=None
                ):
        # 1. Extract features
        bs = 1
        num_cams = img.shape[0]
        feature_maps = self.img_backbone(img)

        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )

        self.img_metas[0]['projection_mat'] = projection_mat
        det_history = {
            'cached_feature': det_cached_feature,
            'cached_anchor': det_cached_anchor,
            'prev_id': det_prev_id,
            'instance_id': det_instance_id,
            'confidence': det_confidence,
            'temp_confidence': det_temp_confidence,
        }

        # pts_bbox_head
        model_outs, det_history_out = self.pts_bbox_head(feature_maps,
                                            self.img_metas,
                                            bank_history=det_history,
                                            time_interval=time_interval,
                                            T_temp2cur=T_temp2cur)

        results = self.pts_bbox_head.post_process(model_outs, self.img_metas)

        # Update detection history
        self.det_histroy['cached_feature'] = det_history_out['cached_feature']
        self.det_histroy['cached_anchor'] = det_history_out['cached_anchor']
        self.det_histroy['prev_id'] = det_history_out['prev_id']
        self.det_histroy['instance_id'] = det_history_out['instance_id']
        self.det_histroy['confidence'] = det_history_out['confidence']
        self.det_histroy['temp_confidence'] = det_history_out['temp_confidence']

        # Update previous meta info
        self.his_timestamp = self.img_metas[0]['timestamp']
        self.his_T_global = self.img_metas[0]['T_global']
        self.his_T_global_inv = self.img_metas[0]['T_global_inv']

        return results, det_history_out


class Sparse4D_export_img_backbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.img_backbone  = model.img_backbone
        self.img_neck      = model.img_neck

    def forward(self, img):
        bs = 1
        num_cams = img.shape[0]
        feature_maps = self.img_backbone(img)

        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )

        return feature_maps


class Sparse4D_export_head(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.depth_branch  = model.depth_branch
        self.pts_bbox_head = model.pts_bbox_head

        self.return_detph = False
        self.use_deformable_func = model.use_deformable_func
        assert self.return_detph is False, "ONNX export does not support depth prediction."
        assert self.use_deformable_func is False, "ONNX export does not support deformable function."

        self.det_histroy = None
        self.det_num_anchor         = model.pts_bbox_head.instance_bank.num_anchor
        self.det_embed_dims         = model.pts_bbox_head.instance_bank.embed_dims
        self.det_num_temp_instances = model.pts_bbox_head.instance_bank.num_temp_instances
        self.anchor_feats_dims  = VZ + 1
       
        self.his_timestamp    = None
        self.his_T_global     = None
        self.his_T_global_inv = None

    def prepare_data(self, img_metas):
        self.img_metas = img_metas
        if self.his_timestamp is None:
            self.his_timestamp = self.img_metas[0]['timestamp']
            self.his_T_global = self.img_metas[0]['T_global']
            self.his_T_global_inv = self.img_metas[0]['T_global_inv']

    def get_history(self, img):
        if self.det_histroy is None:
            bs = 1
            device = img.device

            self.det_histroy = {
                'cached_feature': torch.zeros(bs, self.det_num_temp_instances, self.det_embed_dims).to(device),
                'cached_anchor': torch.zeros(bs, self.det_num_temp_instances, self.anchor_feats_dims).to(device),
                'prev_id': torch.tensor(0).int().to(device),
                'instance_id': -1 * torch.zeros(bs, self.det_num_anchor).int().to(device),
                'confidence': torch.zeros(bs, self.det_num_temp_instances).to(device),
                'temp_confidence': torch.zeros(bs, self.det_num_anchor).to(device),
            }

        history_time  = self.his_timestamp
        time_interval = self.img_metas[0]["timestamp"] - history_time
        T_temp2cur = img.new_tensor(
            np.stack(
                [
                    x["T_global_inv"]
                    @ self.his_T_global
                    for i, x in enumerate(self.img_metas)
                ]
            )
        )

        return time_interval, T_temp2cur

    def forward(self,
                img_feats,
                projection_mat=None,
                det_cached_feature=None,
                det_cached_anchor=None,
                det_prev_id=None,
                det_instance_id=None,
                det_confidence=None,
                det_temp_confidence=None,
                time_interval=None,
                T_temp2cur=None):

        self.img_metas[0]['projection_mat'] = projection_mat
        det_history = {
            'cached_feature': det_cached_feature,
            'cached_anchor': det_cached_anchor,
            'prev_id': det_prev_id,
            'instance_id': det_instance_id,
            'confidence': det_confidence,
            'temp_confidence': det_temp_confidence,
        }

        # pts_bbox_head
        model_outs, det_history_out = self.pts_bbox_head(img_feats,
                                            self.img_metas,
                                            bank_history=det_history,
                                            time_interval=time_interval,
                                            T_temp2cur=T_temp2cur)

        results = self.pts_bbox_head.post_process(model_outs, self.img_metas)

        # Update detection history
        self.det_histroy['cached_feature'] = det_history_out['cached_feature']
        self.det_histroy['cached_anchor'] = det_history_out['cached_anchor']
        self.det_histroy['prev_id'] = det_history_out['prev_id']
        self.det_histroy['instance_id'] = det_history_out['instance_id']
        self.det_histroy['confidence'] = det_history_out['confidence']
        self.det_histroy['temp_confidence'] = det_history_out['temp_confidence']

        # Update previous meta info
        self.his_timestamp = self.img_metas[0]['timestamp']
        self.his_T_global = self.img_metas[0]['T_global']
        self.his_T_global_inv = self.img_metas[0]['T_global_inv']

        return results, det_history_out