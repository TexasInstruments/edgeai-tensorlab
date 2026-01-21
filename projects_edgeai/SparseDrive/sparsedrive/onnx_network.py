import copy
import torch
import torch.nn as nn
import numpy as np

import torchvision.transforms.functional as tF
import torchvision.transforms._functional_tensor as tF_t

#from mmengine.utils import digit_version
#from mmengine.utils.dl_utils import TORCH_VERSION



class SparseDrive_export_model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.img_backbone  = model.img_backbone
        self.img_neck      = model.img_neck
        self.depth_branch  = model.depth_branch
        self.pts_bbox_head = model.pts_bbox_head
        self.img_metas     = None

        self.return_detph = False
        self.use_deformable_func = model.use_deformable_func

        self.det_histroy = None
        self.map_history = None
        self.motion_history = None

        self.det_num_anchor         = model.pts_bbox_head.det_head.instance_bank.num_anchor
        self.det_embed_dims         = model.pts_bbox_head.det_head.instance_bank.embed_dims
        self.det_num_temp_instances = model.pts_bbox_head.det_head.instance_bank.num_temp_instances
        self.map_num_anchor         = model.pts_bbox_head.map_head.instance_bank.num_anchor
        self.map_embed_dims         = model.pts_bbox_head.map_head.instance_bank.embed_dims
        self.map_num_temp_instances = model.pts_bbox_head.map_head.instance_bank.num_temp_instances
        self.num_anchor_feats       = 11
        self.num_ego_feats          = 10

        self.queue_length           = model.pts_bbox_head.motion_plan_head.instance_queue.queue_length

        self.his_timestamp = None
        self.his_T_global = None
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
                'cached_anchor': torch.zeros(bs, self.det_num_temp_instances, self.num_anchor_feats).to(device),
                'prev_id': torch.tensor(0).int().to(device),
                'instance_id': -1 * torch.zeros(bs, self.det_num_anchor).int().to(device),
                'confidence': torch.zeros(bs, self.det_num_temp_instances).to(device),
                'temp_confidence': torch.zeros(bs, self.det_num_anchor).to(device),
            }

            self.map_histroy = {
                'cached_feature': torch.zeros(bs, self.map_num_temp_instances, self.map_embed_dims).to(device),
                'cached_anchor': torch.zeros(bs, self.map_num_temp_instances, 40).to(device),
                'prev_id': torch.tensor(0).int().to(device),
                'instance_id': -1 * torch.zeros(bs, self.map_num_anchor).int().to(device),
                'confidence': torch.zeros(bs, self.map_num_temp_instances).to(device),
                'temp_confidence': torch.zeros(bs, self.map_num_anchor).to(device),
            }

            self.motion_history = {
                'prev_instance_id': -1 * torch.zeros(bs, self.det_num_anchor).int().to(device),
                'prev_confidence': torch.zeros(bs, self.det_num_anchor).to(device),
                'period': torch.zeros(bs, self.det_num_anchor).to(device),
                'instance_feature_queue': torch.zeros(1, bs, self.det_num_anchor, self.det_embed_dims).to(device),
                'anchor_queue': torch.zeros(1, bs, self.det_num_anchor, self.num_anchor_feats).to(device),
                'prev_ego_status': torch.zeros(bs, 1, self.num_ego_feats).to(device),
                'ego_period': torch.tensor(0).int().to(device),
                'ego_feature_queue': torch.zeros(1, bs, 1, self.det_embed_dims).to(device),
                'ego_anchor_queue': torch.zeros(1, bs, 1, self.num_anchor_feats).to(device),
            }

        time_interval = self.img_metas[0]["timestamp"] - self.his_timestamp
        time_interval = torch.Tensor([time_interval]).to(img.device)
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
                imgs,
                projection_mat=None,
                gt_ego_fut_cmd=None,
                det_cached_feature=None,
                det_cached_anchor=None,
                det_prev_id=None,
                det_instance_id=None,
                det_confidence=None,
                det_temp_confidence=None,
                map_cached_feature=None,
                map_cached_anchor=None,
                map_prev_id=None,
                map_instance_id=None,
                map_confidence=None,
                map_temp_confidence=None,
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
                T_temp2cur=None,
                ):
        """
        extract_img_feats()
        """
        B = 1
        #num_cams = imgs.size(0)
        img_feats = self.img_backbone(imgs)
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        assert self.use_deformable_func is False, "ONNX export does not support deformable function."
        assert self.return_detph is False, "ONNX export does not support depth prediction."

        image_wh = []
        timestamp = []
        for x in self.img_metas:
            image_wh.append(x["image_wh"])
            timestamp.append(torch.DoubleTensor([x["timestamp"]]))
        image_wh = torch.stack(image_wh, dim=0).to(imgs.device)
        timestamp = torch.cat(timestamp, dim=0).to(imgs.device)
        data = {
            'return_loss': False,
            'rescale': True,
            'img_metas': self.img_metas,
            'timestamp': timestamp,
            'projection_mat': projection_mat,
            'image_wh': image_wh,
            #'ego_status': ego_status,
            'gt_ego_fut_cmd': gt_ego_fut_cmd.to(imgs.device),
        }

        det_history = {
            'cached_feature': det_cached_feature,
            'cached_anchor': det_cached_anchor,
            'prev_id': det_prev_id,
            'instance_id': det_instance_id,
            'confidence': det_confidence,
            'temp_confidence': det_temp_confidence,
        }
        map_history = {
            'cached_feature': map_cached_feature,
            'cached_anchor': map_cached_anchor,
            'prev_id': map_prev_id,
            'instance_id': map_instance_id,
            'confidence': map_confidence,
            'temp_confidence': map_temp_confidence,
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
        model_outs = self.pts_bbox_head(img_feats_reshaped, data,
                        det_history=det_history,  map_history=map_history, motion_history=motion_history,
                        time_interval=time_interval, T_temp2cur=T_temp2cur)

        det_output, map_output, motion_output, planning_output, det_history_out, map_history_out, motion_history_out = model_outs
        results = self.pts_bbox_head.post_process((det_output, map_output, motion_output, planning_output), data)

        # Set det, map and mostion history
        self.det_histroy['cached_feature'] = det_history_out['cached_feature']
        self.det_histroy['cached_anchor'] = det_history_out['cached_anchor']
        self.det_histroy['prev_id'] = det_history_out['prev_id']
        self.det_histroy['instance_id'] = det_history_out['instance_id']
        self.det_histroy['confidence'] = det_history_out['confidence']
        self.det_histroy['temp_confidence'] = det_history_out['temp_confidence']

        self.map_histroy['cached_feature'] = map_history_out['cached_feature']
        self.map_histroy['cached_anchor'] = map_history_out['cached_anchor']
        self.map_histroy['prev_id'] = map_history_out['prev_id']
        self.map_histroy['instance_id'] = map_history_out['instance_id']
        self.map_histroy['confidence'] = map_history_out['confidence']
        self.map_histroy['temp_confidence'] = map_history_out['temp_confidence']

        self.motion_history['prev_instance_id'] = motion_history_out['prev_instance_id']
        self.motion_history['prev_confidence'] = motion_history_out['prev_confidence']
        self.motion_history['period'] = motion_history_out['period']
        self.motion_history['instance_feature_queue'] = \
            torch.cat((self.motion_history['instance_feature_queue'], motion_history_out['instance_feature_queue']), dim=0)
        self.motion_history['anchor_queue'] = \
            torch.cat((self.motion_history['anchor_queue'], motion_history_out['anchor_queue']), dim=0)
        self.motion_history['prev_ego_status'] = motion_history_out['prev_ego_status']
        self.motion_history['ego_period'] = motion_history_out['ego_period']
        self.motion_history['ego_feature_queue'] = \
            torch.cat((self.motion_history['ego_feature_queue'], motion_history_out['ego_feature_queue']), dim=0)
        self.motion_history['ego_anchor_queue'] = \
            torch.cat((self.motion_history['ego_anchor_queue'], motion_history_out['ego_anchor_queue']), dim=0)

        # Put the first queue elemement if the queue length exceed the limit
        if self.motion_history['instance_feature_queue'].shape[0] > self.queue_length-1:
            self.motion_history['instance_feature_queue'] = self.motion_history['instance_feature_queue'][1:]
            self.motion_history['anchor_queue'] = self.motion_history['anchor_queue'][1:]
            self.motion_history['ego_feature_queue'] = self.motion_history['ego_feature_queue'][1:]
            self.motion_history['ego_anchor_queue'] = self.motion_history['ego_anchor_queue'][1:]

        return results, det_history_out, map_history_out, motion_history_out


class SparseDrive_export_img_backbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.img_backbone  = model.img_backbone
        self.img_neck      = model.img_neck

    def forward(self, imgs):
        B = 1
        img_feats = self.img_backbone(imgs)
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped


class SparseDrive_export_pts_bbox_head(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.depth_branch  = model.depth_branch
        self.pts_bbox_head = model.pts_bbox_head
        self.data          = None

        self.return_detph = False
        self.use_deformable_func = model.use_deformable_func

        self.det_histroy = None
        self.map_history = None
        self.motion_history = None

        self.det_num_anchor         = model.pts_bbox_head.det_head.instance_bank.num_anchor
        self.det_embed_dims         = model.pts_bbox_head.det_head.instance_bank.embed_dims
        self.det_num_temp_instances = model.pts_bbox_head.det_head.instance_bank.num_temp_instances
        self.map_num_anchor         = model.pts_bbox_head.map_head.instance_bank.num_anchor
        self.map_embed_dims         = model.pts_bbox_head.map_head.instance_bank.embed_dims
        self.map_num_temp_instances = model.pts_bbox_head.map_head.instance_bank.num_temp_instances
        self.num_anchor_feats       = 11
        self.num_ego_feats          = 10

        self.queue_length           = model.pts_bbox_head.motion_plan_head.instance_queue.queue_length

        self.his_timestamp = None
        self.his_T_global = None
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
                'cached_anchor': torch.zeros(bs, self.det_num_temp_instances, self.num_anchor_feats).to(device),
                'prev_id': torch.tensor(0).int().to(device),
                'instance_id': -1 * torch.zeros(bs, self.det_num_anchor).int().to(device),
                'confidence': torch.zeros(bs, self.det_num_temp_instances).to(device),
                'temp_confidence': torch.zeros(bs, self.det_num_anchor).to(device),
            }

            self.map_histroy = {
                'cached_feature': torch.zeros(bs, self.map_num_temp_instances, self.map_embed_dims).to(device),
                'cached_anchor': torch.zeros(bs, self.map_num_temp_instances, 40).to(device),
                'prev_id': torch.tensor(0).int().to(device),
                'instance_id': -1 * torch.zeros(bs, self.map_num_anchor).int().to(device),
                'confidence': torch.zeros(bs, self.map_num_temp_instances).to(device),
                'temp_confidence': torch.zeros(bs, self.map_num_anchor).to(device),
            }

            self.motion_history = {
                'prev_instance_id': -1 * torch.zeros(bs, self.det_num_anchor).int().to(device),
                'prev_confidence': torch.zeros(bs, self.det_num_anchor).to(device),
                'period': torch.zeros(bs, self.det_num_anchor).to(device),
                'instance_feature_queue': torch.zeros(1, bs, self.det_num_anchor, self.det_embed_dims).to(device),
                'anchor_queue': torch.zeros(1, bs, self.det_num_anchor, self.num_anchor_feats).to(device),
                'prev_ego_status': torch.zeros(bs, 1, self.num_ego_feats).to(device),
                'ego_period': torch.tensor(0).int().to(device),
                'ego_feature_queue': torch.zeros(1, bs, 1, self.det_embed_dims).to(device),
                'ego_anchor_queue': torch.zeros(1, bs, 1, self.num_anchor_feats).to(device),
            }

        time_interval = self.img_metas[0]["timestamp"] - self.his_timestamp
        time_interval = torch.Tensor([time_interval]).to(img.device)
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
                gt_ego_fut_cmd=None,
                det_cached_feature=None,
                det_cached_anchor=None,
                det_prev_id=None,
                det_instance_id=None,
                det_confidence=None,
                det_temp_confidence=None,
                map_cached_feature=None,
                map_cached_anchor=None,
                map_prev_id=None,
                map_instance_id=None,
                map_confidence=None,
                map_temp_confidence=None,
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
                T_temp2cur=None,
                ):
        assert self.use_deformable_func is False, "ONNX export does not support deformable function."
        assert self.return_detph is False, "ONNX export does not support depth prediction."

        image_wh = []
        timestamp = []
        for x in self.img_metas:
            image_wh.append(x["image_wh"])
            timestamp.append(torch.DoubleTensor([x["timestamp"]]))
        image_wh = torch.stack(image_wh, dim=0).to(img_feats[0].device)
        timestamp = torch.cat(timestamp, dim=0).to(img_feats[0].device)
        data = {
            'return_loss': False,
            'rescale': True,
            'img_metas': self.img_metas,
            'timestamp': timestamp,
            'projection_mat': projection_mat,
            'image_wh': image_wh,
            'gt_ego_fut_cmd': gt_ego_fut_cmd.to(img_feats[0].device),
        }

        det_history = {
            'cached_feature': det_cached_feature,
            'cached_anchor': det_cached_anchor,
            'prev_id': det_prev_id,
            'instance_id': det_instance_id,
            'confidence': det_confidence,
            'temp_confidence': det_temp_confidence,
        }
        map_history = {
            'cached_feature': map_cached_feature,
            'cached_anchor': map_cached_anchor,
            'prev_id': map_prev_id,
            'instance_id': map_instance_id,
            'confidence': map_confidence,
            'temp_confidence': map_temp_confidence,
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
        model_outs = self.pts_bbox_head(img_feats, data,
                        det_history=det_history,  map_history=map_history, motion_history=motion_history,
                        time_interval=time_interval, T_temp2cur=T_temp2cur)

        det_output, map_output, motion_output, planning_output, det_history_out, map_history_out, motion_history_out = model_outs
        results = self.pts_bbox_head.post_process((det_output, map_output, motion_output, planning_output), data)

        # Set det, map and mostion history
        self.det_histroy['cached_feature'] = det_history_out['cached_feature']
        self.det_histroy['cached_anchor'] = det_history_out['cached_anchor']
        self.det_histroy['prev_id'] = det_history_out['prev_id']
        self.det_histroy['instance_id'] = det_history_out['instance_id']
        self.det_histroy['confidence'] = det_history_out['confidence']
        self.det_histroy['temp_confidence'] = det_history_out['temp_confidence']

        self.map_histroy['cached_feature'] = map_history_out['cached_feature']
        self.map_histroy['cached_anchor'] = map_history_out['cached_anchor']
        self.map_histroy['prev_id'] = map_history_out['prev_id']
        self.map_histroy['instance_id'] = map_history_out['instance_id']
        self.map_histroy['confidence'] = map_history_out['confidence']
        self.map_histroy['temp_confidence'] = map_history_out['temp_confidence']

        self.motion_history['prev_instance_id'] = motion_history_out['prev_instance_id']
        self.motion_history['prev_confidence'] = motion_history_out['prev_confidence']
        self.motion_history['period'] = motion_history_out['period']
        self.motion_history['instance_feature_queue'] = \
            torch.cat((self.motion_history['instance_feature_queue'], motion_history_out['instance_feature_queue']), dim=0)
        self.motion_history['anchor_queue'] = \
            torch.cat((self.motion_history['anchor_queue'], motion_history_out['anchor_queue']), dim=0)
        self.motion_history['prev_ego_status'] = motion_history_out['prev_ego_status']
        self.motion_history['ego_period'] = motion_history_out['ego_period']
        self.motion_history['ego_feature_queue'] = \
            torch.cat((self.motion_history['ego_feature_queue'], motion_history_out['ego_feature_queue']), dim=0)
        self.motion_history['ego_anchor_queue'] = \
            torch.cat((self.motion_history['ego_anchor_queue'], motion_history_out['ego_anchor_queue']), dim=0)

        # Pot the first queue elemement if the queue length exceed the limit
        if self.motion_history['instance_feature_queue'].shape[0] > self.queue_length-1:
            self.motion_history['instance_feature_queue'] = self.motion_history['instance_feature_queue'][1:]
            self.motion_history['anchor_queue'] = self.motion_history['anchor_queue'][1:]
            self.motion_history['ego_feature_queue'] = self.motion_history['ego_feature_queue'][1:]
            self.motion_history['ego_anchor_queue'] = self.motion_history['ego_anchor_queue'][1:]

        return results, det_history_out, map_history_out, motion_history_out
