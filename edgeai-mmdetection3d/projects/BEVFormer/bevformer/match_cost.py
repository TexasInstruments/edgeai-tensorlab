import torch
import mmcv
from mmdet3d.registry import TASK_UTILS
from torch import Tensor


@TASK_UTILS.register_module()
class FocalLossCost(object):
    """FocalLossCost.

    Args:
        alpha (Union[float, int]): focal_loss alpha. Defaults to 0.25.
        gamma (Union[float, int]): focal_loss gamma. Defaults to 2.
        eps (float): Defaults to 1e-12.
        binary_input (bool): Whether the input is binary. Currently,
            binary_input = True is for masks input, binary_input = False
            is for label input. Defaults to False.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 eps: float = 1e-12,
                 binary_input: bool = False,
                 weight= 1.) -> None:
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input

    def __call__(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_queries, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight


@TASK_UTILS.register_module()
class BBox3DL1Cost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

#@mmcv.jit(derivate=True, coderize=True)
#@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.
    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    # assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss.sum(-1)


@TASK_UTILS.register_module()
class SmoothL1Cost(object):
    """SmoothL1Cost.
     Args:
         weight (int | float, optional): loss weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, input, target):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: iou_cost value with weight
        """
        N1, C = input.shape
        N2, C = target.shape
        input = input.contiguous().view(N1, C)[:, None, :]
        target = target.contiguous().view(N2, C)[None, :, :]
        cost = smooth_l1_loss(input, target)

        return cost * self.weight