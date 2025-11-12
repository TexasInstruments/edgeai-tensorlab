# Copyright (c) 2018-2025, Texas Instruments
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


import torch
import onnx_graphsurgeon as gs
from torch.onnx import symbolic_helper as sym_help
from torch import Tensor
from . import utils

class ONNXNMSop(torch.autograd.Function):
    """Create onnx::NonMaxSuppression op.

    NMS in mmcv only supports one class with no batch info. This class assists
    in exporting NMS of ONNX's definition.
    """

    @staticmethod
    def forward(ctx, boxes: Tensor, scores: Tensor,
                max_output_boxes_per_class: int, iou_threshold: float,
                score_threshold: float) -> Tensor:
        """Get NMS output indices.

        Args:
            ctx (Context): The context with meta information.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            max_output_boxes_per_class (int): Maximum number of output
                boxes per class of nms.
            iou_threshold (float): IOU threshold of nms.
            score_threshold (float): score threshold of nms.

        Returns:
            Tensor: Selected indices of boxes. 2-D tensor of shape
            (num_selected_indices, 3) with each row of
            [batch_index, class_index, box_index].
        """
        from torchvision.ops.boxes import nms
        batch_size, num_class, _ = scores.shape

        max_output_boxes_per_class = int(max_output_boxes_per_class)
        score_threshold = float(score_threshold)
        iou_threshold = float(iou_threshold)
    
        indices = []
        for batch_id in range(batch_size):
            for cls_id in range(num_class):
                _boxes = boxes[batch_id, ...]
                # score_threshold=0 requires scores to be contiguous
                
                _scores = scores[batch_id, cls_id, ...].contiguous()
                flag = _scores >= score_threshold
                box_inds = nms(
                    _boxes,
                    _scores,
                    iou_threshold,
                    )
                box_inds = box_inds[flag[box_inds]]
                if max_output_boxes_per_class:
                    box_inds = box_inds[:max_output_boxes_per_class]
                
                batch_inds = torch.zeros_like(box_inds) + batch_id
                cls_inds = torch.zeros_like(box_inds) + cls_id
                indices.append(
                    torch.stack([batch_inds, cls_inds, box_inds], dim=-1))
        indices = torch.cat(indices)
        return indices

    @staticmethod
    def symbolic(g, boxes: Tensor, scores: Tensor,
                 max_output_boxes_per_class: int, iou_threshold: float,
                 score_threshold: float):
        """Symbolic function for onnx::NonMaxSuppression.

        Args:
            g (Graph): The traced onnx graph.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            max_output_boxes_per_class (int): Maximum number of output
                boxes per class of nms.
            iou_threshold (float): IOU threshold of nms.
            score_threshold (float): score threshold of nms.

        Returns:
            NonMaxSuppression op for onnx.
        """
        if not sym_help._is_value(max_output_boxes_per_class):
            max_output_boxes_per_class = g.op(
                'Constant',
                value_t=torch.tensor(
                    max_output_boxes_per_class, dtype=torch.long))

        if not sym_help._is_value(iou_threshold):
            iou_threshold = g.op(
                'Constant',
                value_t=torch.tensor([iou_threshold], dtype=torch.float))

        if not sym_help._is_value(score_threshold):
            score_threshold = g.op(
                'Constant',
                value_t=torch.tensor([score_threshold], dtype=torch.float))
        return g.op('NonMaxSuppression', boxes, scores,
                    max_output_boxes_per_class, iou_threshold, score_threshold)

def torch_non_max_suppression(boxes: Tensor, scores: Tensor, max_output_boxes_per_class: int=None, iou_threshold: float=0, score_threshold: float=0, center_point_box = 0) -> Tensor:
    """Get NMS output indices.

    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms.
        iou_threshold (float): IOU threshold of nms.
        score_threshold (float): score threshold of nms.

    Returns:
        Tensor: Selected indices of boxes. 2-D tensor of shape
        (num_selected_indices, 3) with each row of
        [batch_index, class_index, box_index].
    """
    if isinstance(max_output_boxes_per_class, torch.Tensor):
        max_output_boxes_per_class = max_output_boxes_per_class.cpu().tolist()
    if isinstance(max_output_boxes_per_class, (list, tuple)):
        if len(max_output_boxes_per_class) == 1:
            max_output_boxes_per_class = int(max_output_boxes_per_class[0])
        else:
            raise ValueError('max_output_boxes_per_class should be an int or a list/tuple of length 1, but got {}.'.format(max_output_boxes_per_class))
    # if isinstance(iou_threshold, torch.Tensor):
    #     iou_threshold = iou_threshold.cpu().tolist()
    if isinstance(iou_threshold, (list, tuple)) or (isinstance(iou_threshold, torch.Tensor) and iou_threshold.dim() == 1):
        if len(iou_threshold) == 1:
            iou_threshold = float((iou_threshold.detach() if isinstance(iou_threshold, torch.Tensor) else iou_threshold)[0])
        else:
            raise ValueError('iou_threshold should be an int or a list/tuple of length 1, but got {}.'.format(iou_threshold))
    # if isinstance(score_threshold, torch.Tensor):
    #     score_threshold = score_threshold.cpu().tolist()
    if isinstance(score_threshold, (list, tuple))or (isinstance(iou_threshold, torch.Tensor) and iou_threshold.dim() == 1):
        if len(score_threshold) == 1:
            score_threshold = float((score_threshold.detach() if isinstance(score_threshold, torch.Tensor) else score_threshold)[0])
        else:
            raise ValueError('score_threshold should be an int or a list/tuple of length 1, but got {}.'.format(score_threshold))
    return ONNXNMSop.apply(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)

    
def add_non_max_suppression_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    try:
        from torchvision.ops.boxes import nms
    except Exception as e:
        print("Failed to import nms from torchvision.ops.boxes! Please install torchvision and try again!")
        raise e
    assert 2 <= len(node.inputs)<=5, f'{node.name} with operator {node.op} should have between 2 and 5 inputs, but got {len(node.inputs)}'
    types = [torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(node, inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    center_point_box = node.attrs.get('center_point_box', 0)
    if state.module_based:
        module = utils.WrappedModule(node.name, node.op, torch_module, torch_non_max_suppression, args, dict(center_point_box=center_point_box))
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_non_max_suppression, tuple(args), dict(center_point_box=center_point_box), name=node.name)
