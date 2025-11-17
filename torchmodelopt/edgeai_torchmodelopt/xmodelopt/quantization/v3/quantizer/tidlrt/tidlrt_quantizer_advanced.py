
#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################


import copy
from typing import Callable, Optional
import torch
from torch.fx import Node
from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
)
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    SharedQuantizationSpec
)


from ..xnnpack import XNNPACKQuantizer, QuantizationConfig
from ..xnnpack import set_annotation_patterns, register_annotator, AnnotatorType
from ..xnnpack import get_input_act_qspec, get_output_act_qspec, get_weight_qspec, get_bias_qspec
from ..xnnpack import _is_annotated, _mark_nodes_as_annotated, get_annotation_func, _is_input_large_scalar, _is_input_non_float_tensor
from ..xnnpack import OP_TO_ANNOTATOR, STATIC_OPS_BACKUP, STATIC_QAT_ONLY_OPS_BACKUP, DYNAMIC_OPS_BACKUP


def get_aten_overload_ops(aten_op_name: str):
    aten_op_overload_packet = getattr(torch.ops.aten, aten_op_name)
    overload_names = aten_op_overload_packet.overloads()
    op_overloads = []
    for overload_name in overload_names:
        op_overload = getattr(aten_op_overload_packet, overload_name)
        op_overloads += [op_overload]
    #
    return op_overloads


# @register_annotator('matmul')
# def _annotate_matmul(
#     gm: torch.fx.GraphModule,
#     quantization_config: Optional[QuantizationConfig],
#     filter_fn: Optional[Callable[[Node], bool]] = None,
# ) -> Optional[list[list[Node]]]:

#     # matmul is currently not quantized
#     annotated_partitions = []
#     for node in gm.graph.nodes:
#         if node.op != "call_function" or node.target != torch.ops.aten.matmul.default:
#             continue
#         if filter_fn and not filter_fn(node):
#             continue

#         matmul_node = node
#         nodes_to_mark_annotated = [matmul_node]

#         if len(node.users) == 1:
#             next_node = node.users[0]
#             if next_node.op == "call_function" and next_node.target in [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor]:
#                 add_node = next_node
#                 nodes_to_mark_annotated += [add_node]
#                 # add_node.meta["quantization_annotation"] = QuantizationAnnotation(
#                 #         input_qspec_map=input_qspec_map,
#                 #         output_qspec=output_act_qspec,
#                 #         _annotated=True,
#                 #     )
    
#         _mark_nodes_as_annotated(nodes_to_mark_annotated)
#         annotated_partitions.append(nodes_to_mark_annotated)

#     return annotated_partitions


class TIDLRTQuantizerAdvanced(XNNPACKQuantizer):
    # XNNPACKQuantizer.STATIC_OPS.insert(0, 'matmul')
    pass


def get_quantizer(is_qat=True, fast_mode=False, device=None, annotation_patterns=None, **kwargs):
    set_annotation_patterns(annotation_patterns=annotation_patterns)
    return TIDLRTQuantizerAdvanced(**kwargs)

