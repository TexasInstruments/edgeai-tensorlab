import torch

from mmdet.models.task_modules import BaseBBoxCoder
from mmdet3d.registry import TASK_UTILS
from .util import denormalize_bbox
import numpy as np


@TASK_UTILS.register_module()
class NMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):

        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        # for TIDL detection post processing
        if torch.onnx.is_in_onnx_export():
            num_query, num_ch = cls_scores.shape
            sq_num_query = int(np.sqrt(num_query))
            cls_scores = cls_scores.transpose(0, 1)
            cls_scores = cls_scores.reshape(-1, sq_num_query, sq_num_query)
            cls_scores = cls_scores.permute(1, 2, 0)
            cls_scores = cls_scores.reshape(num_query, -1)
            bbox_preds = bbox_preds.transpose(0, 1)
            bbox_preds = bbox_preds.reshape(-1, sq_num_query, sq_num_query)
            bbox_preds = bbox_preds.permute(1, 2, 0)
            bbox_preds = bbox_preds.reshape(num_query, -1)

        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            #mask &= (final_box_preds[..., :3] <=
            #         self.post_center_range[3:]).all(1)
            mask = mask & ((final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1))

            if self.score_threshold:
                #mask &= thresh_mask
                mask = mask & thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]

            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        if torch.onnx.is_in_onnx_export():
            all_cls_scores = preds_dicts['all_cls_scores'][-1:].squeeze(0)
            all_bbox_preds = preds_dicts['all_bbox_preds'][-1:].squeeze(0)
        else:
            all_cls_scores = preds_dicts['all_cls_scores'][-1]
            all_bbox_preds = preds_dicts['all_bbox_preds'][-1]

        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            if torch.onnx.is_in_onnx_export() and batch_size == 1:
                predictions_list.append(self.decode_single(all_cls_scores.squeeze(0), all_bbox_preds.squeeze(0)))
            else:
                predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list

