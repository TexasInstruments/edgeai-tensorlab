import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple

from torchvision.ops import boxes as box_ops
from torchvision.ops.boxes import nms

from torchvision.models.detection.roi_heads import paste_masks_in_image
from torchvision.models.detection.transform import resize_boxes, resize_keypoints
from torchvision.models.detection import _utils as det_utils


def _batched_nms_coordinate_tricks_custom(boxes, scores, idxs, iou_threshold):
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def postprocess_detections_ssd(
    self, head_outputs: Dict[str, Tensor], image_anchors: List[Tensor], image_shapes: List[Tuple[int, int]]
) -> List[Dict[str, Tensor]]:
    bbox_regression = head_outputs["bbox_regression"]
    pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1)

    num_classes = pred_scores.size(-1)
    device = pred_scores.device

    detections: List[Dict[str, Tensor]] = []

    for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
        boxes = self.box_coder.decode_single(boxes, anchors)
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        image_boxes = []
        image_scores = []
        image_labels = []
        for label in range(1, num_classes):
            score = scores[:, label]

            keep_idxs = score > self.score_thresh
            score = score[keep_idxs]
            box = boxes[keep_idxs]

            # keep only topk scoring predictions
            num_topk = det_utils._topk_min(score, self.topk_candidates, 0)
            score, idxs = score.topk(num_topk)
            box = box[idxs]

            image_boxes.append(box)
            image_scores.append(score)
            image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)

        # non-maximum suppression
        keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
        keep = keep[: self.detections_per_img]

        image_dets = torch.cat([image_boxes[keep], image_scores[keep].unsqueeze(1)], dim=1)

        detections.append(
            {
                "dets": image_dets,
                "labels": image_labels[keep],
            }
        )
    return detections

def postprocess_ssd(
        self,
        result: List[Dict[str, Tensor]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["dets"][:,:4]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["dets"][:,:4] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result