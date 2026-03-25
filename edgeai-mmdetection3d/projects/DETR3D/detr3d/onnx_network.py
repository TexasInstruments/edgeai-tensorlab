from typing import Dict, List
import torch
import torch.nn as nn

from mmdet3d.structures.bbox_3d.utils import get_lidar2img



class DETR3D_export_model(nn.Module):
    def __init__(self,
                 img_backbone,
                 img_neck,
                 pts_bbox_head,
                 add_pred_to_datasample):
        super().__init__()
        self.img_backbone           = img_backbone
        self.img_neck               = img_neck
        self.pts_bbox_head          = pts_bbox_head
        self.add_pred_to_datasample = add_pred_to_datasample


    def add_lidar2img(self, batch_input_metas: List[Dict]) -> List[Dict]:
        """add 'lidar2img' transformation matrix into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.
        Returns:
            batch_input_metas (list[dict]): Meta info with lidar2img added
        """
        for meta in batch_input_metas:
            l2i = list()
            for i in range(len(meta['cam2img'])):
                c2i = torch.tensor(meta['cam2img'][i]).double()
                l2c = torch.tensor(meta['lidar2cam'][i]).double()
                l2i.append(get_lidar2img(c2i, l2c).float().numpy())
            meta['lidar2img'] = l2i

        return batch_input_metas

        """
        for meta in batch_input_metas:
            lidar2img_rts = []
            # obtain lidar to image transformation matrix
            for i in range(len(meta['cam2img'])):
                lidar2cam_rt = torch.tensor(meta['lidar2cam'][i]).double()
                intrinsic = torch.tensor(meta['cam2img'][i]).double()
                viewpad = torch.eye(4).double()
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                # The extrinsics mean the transformation from lidar to camera.
                # If anyone want to use the extrinsics as sensor to lidar,
                # please use np.linalg.inv(lidar2cam_rt.T)
                # and modify the ResizeCropFlipImage
                # and LoadMultiViewImageFromMultiSweepsFiles.
                lidar2img_rts.append(lidar2img_rt)
            meta['lidar2img'] = lidar2img_rts
            img_shape = meta['img_shape'][:3]
            meta['img_shape'] = [img_shape] * len(img[0])

        return batch_input_metas
        """

    def prepare_data(self, img, img_metas):
        self.img_metas = img_metas

        input_shape = img.shape[-2:]
        # update real input shae of each single img
        for img_meta in self.img_metas:
            img_meta.update(input_shape=input_shape)


    #def forward(self, img, batch_data_samples):
    def forward(self, img):
        B, N, C, H, W = img.size()

        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)

        img_feats = self.img_backbone(img)      
        img_feats = self.img_neck(img_feats)
        
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        
        outs = self.pts_bbox_head(img_feats_reshaped, self.img_metas)

        return outs

        # post-processing
        '''        
        results_list_3d = self.pts_bbox_head.predict_by_feat(
            outs, self.img_metas)
        return results_list_3d

        detsamples = self.add_pred_to_datasample(
            batch_data_samples, results_list_3d)

        return detsamples
        '''
