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
        self.head          = model.head
        self.img_metas     = None

        self.return_detph = False
        self.use_deformable_func = model.use_deformable_func


    def prepare_data(self, img_metas):
        self.img_metas = img_metas


    def forward(self,
                imgs,
                timestamp=None,
                projection_mat=None,
                image_wh=None,
                ego_status=None,
                gt_ego_fut_cmd=None):
        """
        extract_img_feats()
        """
        B = 1
        num_cams = imgs.size(0)
        img_feats = self.img_backbone(imgs)
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        assert self.use_deformable_func is False, "ONNX export does not support deformable function."
        assert self.return_detph is False, "ONNX export does not support depth prediction."

        data = {
            'return_loss': False,
            'rescale': True,
            'img_metas': self.img_metas,
            'timestamp': timestamp,
            'projection_mat': projection_mat,
            'image_wh': image_wh,
            'ego_status': ego_status,
            'gt_ego_fut_cmd': gt_ego_fut_cmd
        }
        model_outs = self.head(img_feats_reshaped, data)
        return model_outs
        results = self.head.post_process(model_outs, data)

        return results

