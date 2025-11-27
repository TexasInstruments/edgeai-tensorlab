
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
import itertools
import torch
import torch.nn.functional as F
from torch.fx import Node
from torch.ao.quantization.pt2e.utils import (
    _get_aten_graph_module_for_pattern,
    _is_conv_node,
    _is_conv_transpose_node,
)
from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
)
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    SharedQuantizationSpec
)
from torch.ao.quantization.pt2e.export_utils import _WrapperModule
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap,
)
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


from ..xnnpack import XNNPACKQuantizer, QuantizationConfig
from ..xnnpack import set_annotation_patterns, register_annotator, AnnotatorType, extend_annotation_patterns
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


def _do_annotate_conv_mul_add(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]],
    has_relu: bool,
    is_conv_transpose: bool = False,
) -> list[list[Node]]:
    """
    Given a function that takes in a `conv_fn` and returns a conv-bn[-relu] pattern,
    return a list of annotated partitions.

    The output of the pattern must include a dictionary from string name to node
    for the following names: "input", "conv", "weight", "bias", and "output".
    """

    # Example inputs for conv-bn1d patterns
    _conv1d_mul_add_example_inputs = (
        torch.randn(1, 1, 3),  # x
        torch.randn(1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
        torch.randn(1),  # mul
        torch.randn(1),  # add
    )

    # Example inputs for conv-bn2d patterns
    _conv2d_mul_add_example_inputs = (
        torch.randn(1, 1, 3, 3),  # x
        torch.randn(1, 1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
        torch.randn(1),  # mul
        torch.randn(1),  # add
    )

    def get_pattern(conv_fn: Callable, relu_is_inplace: bool):
        def _conv_mul_add(x, conv_weight, conv_bias, mul, add):
            conv = conv_fn(x, conv_weight, conv_bias)
            mul_add = conv * mul + add
            if has_relu:
                output = F.relu_(mul_add) if relu_is_inplace else F.relu(mul_add)
            else:
                output = mul_add
            return output, {
                "input": x,
                "conv": conv,
                "weight": conv_weight,
                "bias": conv_bias,
                "output": output,
            }

        return _WrapperModule(_conv_mul_add)

    # Needed for matching, otherwise the matches gets filtered out due to unused
    # nodes returned by batch norm
    gm.graph.eliminate_dead_code()
    gm.recompile()

    matches = []
    if is_conv_transpose:
        combinations = [
            (F.conv_transpose1d, _conv1d_mul_add_example_inputs),
            (F.conv_transpose2d, _conv2d_mul_add_example_inputs),
        ]
    else:
        combinations = [
            (F.conv1d, _conv1d_mul_add_example_inputs),  # type: ignore[list-item]
            (F.conv2d, _conv2d_mul_add_example_inputs),  # type: ignore[list-item]
        ]

    # Add `is_cuda` and `relu_is_inplace` dimensions
    combinations = itertools.product(  # type: ignore[assignment]
        combinations,
        [True, False] if torch.cuda.is_available() else [False],  # is_cuda
        [True, False] if has_relu else [False],  # relu_is_inplace
    )

    # Match against all conv dimensions and cuda variants
    for (conv_fn, example_inputs), is_cuda, relu_is_inplace in combinations:  # type: ignore[misc]
        pattern = get_pattern(conv_fn, relu_is_inplace)  # type: ignore[has-type]
        pattern = _get_aten_graph_module_for_pattern(pattern, example_inputs, is_cuda)  # type: ignore[has-type]
        pattern.graph.eliminate_dead_code()
        pattern.recompile()
        matcher = SubgraphMatcherWithNameNodeMap(pattern, ignore_literals=True)
        matches.extend(matcher.match(gm.graph))

    # Annotate nodes returned in the matches
    annotated_partitions = []
    for match in matches:
        name_node_map = match.name_node_map
        input_node = name_node_map["input"]
        conv_node = name_node_map["conv"]
        weight_node = name_node_map["weight"]
        bias_node = name_node_map["bias"]
        output_node = name_node_map["output"]

        # TODO: annotate the uses of input, weight, and bias separately instead
        # of assuming they come from a single conv node. This is not possible today
        # because input may have multiple users, and we can't rely on the conv node
        # always being the first user. This was the case in models with skip
        # connections like resnet18

        # Validate conv args
        if conv_node.args[0] is not input_node:
            raise ValueError("Conv arg did not contain input node ", input_node)
        if conv_node.args[1] is not weight_node:
            raise ValueError("Conv arg did not contain weight node ", weight_node)
        if len(conv_node.args) > 2 and conv_node.args[2] is not bias_node:
            raise ValueError("Conv arg did not contain bias node ", bias_node)

        # Skip if the partition is already annotated or is filtered out by the user
        partition = [conv_node, weight_node]
        if bias_node is not None:
            partition.append(bias_node)
        if _is_annotated(partition):
            continue
        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        # Annotate conv inputs and pattern output
        input_qspec_map = {}
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
        if bias_node is not None:
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        output_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
            _annotated=True,
        )
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("conv_mul_add_relu")
def _annotate_conv_mul_add_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    """
    Find conv_transpose + batchnorm + relu partitions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    return _do_annotate_conv_mul_add(
        gm, quantization_config, filter_fn, has_relu=True, is_conv_transpose=False
    )


@register_annotator('matmul')
def _annotate_matmul(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:

    # matmul is currently not quantized
    # quantizing it needs carefull handling of attention layers
    annotated_partitions = []
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

    return annotated_partitions


class TIDLRTQuantizerAdvanced(XNNPACKQuantizer):
    extend_annotation_patterns(['conv_mul_add_relu', 'matmul'])

    # there is a bug in this transformation in XNNPACKQuantizer - it is not respecting dtype
    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        """Transforms scalar values to tensor attributes"""
        # return _convert_scalars_to_attrs(model)
        return model
    
    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        return super().annotate(model)

def get_quantizer(is_qat=True, fast_mode=False, device=None, annotation_patterns=None, **kwargs):
    set_annotation_patterns(annotation_patterns=annotation_patterns)
    return TIDLRTQuantizerAdvanced(**kwargs)

