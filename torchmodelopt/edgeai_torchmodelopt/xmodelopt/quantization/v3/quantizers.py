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

import torch
from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    get_input_act_qspec,
    get_output_act_qspec,
    get_bias_qspec,
    get_weight_qspec,
    OperatorConfig,
    QuantizationConfig,
)
from torch.ao.quantization.quantizer.quantizer import (
    Quantizer,
    QuantizationAnnotation,
    SharedQuantizationSpec,
    QuantizationSpec
)
import itertools
import operator
from typing import Dict, List, Optional, Any

from torch.fx import Node

from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from . import qconfig_types

def _mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if node is not None:
            if "quantization_annotation" not in node.meta:
                node.meta["quantization_annotation"] = QuantizationAnnotation()
            node.meta["quantization_annotation"]._annotated = True


def _is_annotated(nodes: List[Node]):
    annotated = False
    for node in nodes:
        annotated = annotated or (
            "quantization_annotation" in node.meta
            and node.meta["quantization_annotation"]._annotated
        )
    return annotated


def is_positive_function_present(node, find_level):
    if node.target in (torch.ops.aten._softmax.default, torch.ops.aten.relu.default, torch.ops.aten.sigmoid.default) :
        return True
    elif find_level>0:
        return is_positive_function_present(node.args[0], find_level-1)
    else:
        return False
    
   
def is_mlp_add_layer(node, find_level, found_linear=False, linear_node=None):
    if find_level < 0 or node is None :
        return False, linear_node
    elif node.target == torch.ops.aten.gelu.default:
        if found_linear:
            return True, linear_node
        else:
            # found gelu before the linear layer
            return False, linear_node
        #
    elif node.target == torch.ops.aten.addmm.default:
        found_linear = True
        linear_node = node
        #
    #
    prev_node = None
    for n_id in node.args:
        if isinstance(n_id, Node) and n_id.op != 'get_attr':
            prev_node = n_id
            break
    return is_mlp_add_layer(prev_node, find_level-1, found_linear, linear_node)


class TIDLRTQuantizer(Quantizer):

    def __init__(self):
        super().__init__()
        self.global_config: QuantizationConfig = None  # type: ignore[assignment]
        self.operator_type_config: Dict[str, Optional[QuantizationConfig]] = {}

    def set_global(self, quantization_config: QuantizationConfig):
        """set global QuantizationConfig used for the backend.
        QuantizationConfig is defined in torch/ao/quantization/_pt2e/quantizer/quantizer.py.
        """
        self.global_config = quantization_config
        return self

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """annotate nodes in the graph with observer or fake quant constructors
        to convey the desired way of quantization.
        """
        global_config = self.global_config
        self.annotate_config(model, global_config)

        return model

    def annotate_config(
        self, model: torch.fx.GraphModule, config: QuantizationConfig, allow_16bit_node_list: list = []
    ) -> torch.fx.GraphModule:
        # allow_16bit_node_list : TODO
        # quantize the weight of that layer as well as the output to 16 bit, however, input is still 8 bit quantized
        # further, the quantization also flows, which means, if the input is 16 bit, then weights will also be in 16 bit
        # but the output will be in 8 bit
        self._annotate_linear(model, config, allow_16bit_node_list)
        self._annotate_conv2d(model, config, allow_16bit_node_list)
        self._annotate_maxpool2d(model, config)
        self._annotate_softmax(model, config)
        self._annotate_matmul(model, config)
        self._annotate_layernorm(model, config)
        self._annotate_cat(model, config)
        self._annotate_mul(model, config)
        self._annotate_add(model, config)
        return model

    def _annotate_conv2d(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig, allow_16bit_node_list: list
    ) -> None:
        conv_partitions = get_source_partitions(
            gm.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d]
        )
        conv_partitions = list(itertools.chain(*conv_partitions.values()))
        for conv_partition in conv_partitions:
            if len(conv_partition.output_nodes) > 1:
                raise ValueError("conv partition has more than one output node")
            conv_node = conv_partition.output_nodes[0]
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.convolution.default
            ):
                raise ValueError(f"{conv_node} is not an aten conv2d operator")
            # skip annotation if it is already annotated
            if _is_annotated([conv_node]):
                continue

            input_qspec_map = {}
            input_act = conv_node.args[0]
            assert isinstance(input_act, Node)
            input_qspec_map[input_act] = get_input_act_qspec(quantization_config)

            weight = conv_node.args[1]
            assert isinstance(weight, Node)
            input_qspec_map[weight] = get_weight_qspec(quantization_config)

            bias = conv_node.args[2]
            if isinstance(bias, Node):
                input_qspec_map[bias] = get_bias_qspec(quantization_config)

            conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )

    def _annotate_mul(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        mul_partitions = get_source_partitions(
            gm.graph, [torch.mul, operator.mul])
        mul_partitions = list(itertools.chain(*mul_partitions.values()))
        for mul_partition in mul_partitions:
            mul_node = mul_partition.output_nodes[0]
            if (
                mul_node.op != "call_function"
                or mul_node.target != torch.ops.aten.mul.Tensor
            ):
                raise ValueError(f"{mul_node} is not an aten mul operator")
            if _is_annotated([mul_node]):
                continue
            input_qspec_map = {}
            input_act = mul_node.args[0]
            assert isinstance(input_act, Node)
            input_qspec_map[input_act] = get_input_act_qspec(quantization_config)
            bias = mul_node.args[1]
            if isinstance(bias, Node):
                input_qspec_map[bias] = get_bias_qspec(quantization_config)
            mul_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )
        
    def _annotate_linear(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig, allow_16bit_node_list: list
    ) -> None:
        module_partitions = get_source_partitions(
            gm.graph, [torch.nn.Linear, torch.nn.functional.linear]
        )
        act_qspec = get_input_act_qspec(quantization_config)
        weight_qspec = get_weight_qspec(quantization_config)
        bias_qspec = get_bias_qspec(quantization_config)
        for module_or_fn_type, partitions in module_partitions.items():
            if module_or_fn_type == torch.nn.Linear:
                for p in partitions:
                    act_node = p.input_nodes[0]
                    output_node = p.output_nodes[0]
                    weight_node = None
                    bias_node = None
                    for node in p.params:
                        weight_or_bias = getattr(gm, node.target)  # type: ignore[arg-type]
                        if weight_or_bias.ndim == 2:  # type: ignore[attr-defined]
                            weight_node = node
                        if weight_or_bias.ndim == 1:  # type: ignore[attr-defined]
                            bias_node = node
                    if weight_node is None:
                        raise ValueError("No weight found in Linear pattern")
                    # find use of act node within the matched pattern
                    act_use_node = None
                    for node in p.nodes:
                        if node in act_node.users:  # type: ignore[union-attr]
                            act_use_node = node
                            break
                    if act_use_node is None:
                        raise ValueError(
                            "Could not find an user of act node within matched pattern."
                        )
                    if _is_annotated([act_use_node]) is False:  # type: ignore[list-item]
                        _annotate_input_qspec_map(
                            act_use_node,
                            act_node,
                            act_qspec,
                        )
                    if bias_node and _is_annotated([bias_node]) is False:
                        _annotate_output_qspec(bias_node, bias_qspec)
                    if _is_annotated([weight_node]) is False:  # type: ignore[list-item]
                        _annotate_output_qspec(weight_node, weight_qspec)
                    if _is_annotated([output_node]) is False:
                        _annotate_output_qspec(output_node, act_qspec)
                    nodes_to_mark_annotated = list(p.nodes)
                    _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_maxpool2d(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        module_partitions = get_source_partitions(
            gm.graph, [torch.nn.MaxPool2d, torch.nn.functional.max_pool2d]
        )
        maxpool_partitions = list(itertools.chain(*module_partitions.values()))
        for maxpool_partition in maxpool_partitions:
            output_node = maxpool_partition.output_nodes[0]
            maxpool_node = None
            for n in maxpool_partition.nodes:
                if n.target == torch.ops.aten.max_pool2d_with_indices.default:
                    maxpool_node = n
            if _is_annotated([output_node, maxpool_node]):  # type: ignore[list-item]
                continue

            input_act = maxpool_node.args[0]  # type: ignore[union-attr]
            assert isinstance(input_act, Node)

            act_qspec = get_input_act_qspec(quantization_config)
            maxpool_node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
                input_qspec_map={
                    input_act: act_qspec,
                },
                _annotated=True,
            )
            output_node.meta["quantization_annotation"] = QuantizationAnnotation(
                output_qspec=SharedQuantizationSpec((input_act, maxpool_node)),
                _annotated=True,
            )
            
    def _annotate_softmax(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        module_partitions = get_source_partitions(
            gm.graph, [torch.nn.Softmax, torch.nn.functional.softmax]
        )
        softmax_partitions = list(itertools.chain(*module_partitions.values()))
        for softmax_partition in softmax_partitions:
            output_node = softmax_partition.output_nodes[0]
            softmax_node = None
            for n in softmax_partition.nodes:
                if n.target == torch.ops.aten._softmax.default:
                    softmax_node = n
            if _is_annotated([output_node, softmax_node]):  # type: ignore[list-item]
                continue

            input_act = softmax_node.args[0]  # type: ignore[union-attr]
            assert isinstance(input_act, Node)

            act_qspec = get_input_act_qspec(quantization_config)
            softmax_node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
                input_qspec_map={
                    input_act: act_qspec,
                },
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )
            
    def _annotate_matmul(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        module_partitions = get_source_partitions(
            gm.graph, [torch.matmul, torch.bmm, operator.matmul]
        )
        # TODO take care of the bias term from bmm
        matmul_partitions = list(itertools.chain(*module_partitions.values()))
        for matmul_partition in matmul_partitions:
            output_node = matmul_partition.output_nodes[0]
            matmul_node = None
            for n in matmul_partition.nodes:
                if n.target == torch.ops.aten.bmm.default:
                    matmul_node = n
            if _is_annotated([output_node, matmul_node]):  # type: ignore[list-item]
                continue
            
            act_qspec = get_input_act_qspec(quantization_config)
            # setting the symmetric inputs
            # messy way of doing it, need better #TODO
            act_qspec_symmetric = qconfig_types.get_act_quantization_config(
                dict(
                    qscheme=torch.per_tensor_symmetric, 
                    power2_scale=act_qspec.observer_or_fake_quant_ctr.p.keywords['observer'].__init__._partialmethod.keywords['power2_scale'], 
                    range_shrink_percentile=act_qspec.observer_or_fake_quant_ctr.p.keywords['observer'].__init__._partialmethod.keywords['range_shrink_percentile']
                )
            )

            input_act0 = matmul_node.args[0]  # type: ignore[union-attr]
            assert isinstance(input_act0, Node)
            
            input_act1 = matmul_node.args[1]  # type: ignore[union-attr]
            assert isinstance(input_act1, Node)
            
            if is_positive_function_present(input_act0, 3): 
                input_act0_spec = act_qspec
                input_act1_spec = act_qspec_symmetric
            elif is_positive_function_present(input_act1, 3):
                input_act1_spec = act_qspec
                input_act0_spec = act_qspec_symmetric
            else:
                input_act1_spec = act_qspec_symmetric
                input_act0_spec = act_qspec_symmetric
            
            matmul_node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
                input_qspec_map={
                    input_act0: input_act0_spec,
                    input_act1: input_act1_spec,
                },
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )

    def _annotate_layernorm(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        module_partitions = get_source_partitions(
            gm.graph, [torch.nn.LayerNorm, torch.nn.functional.layer_norm]
        )
        layernorm_partitions = list(itertools.chain(*module_partitions.values()))
        for layernorm_partition in layernorm_partitions:
            output_node = layernorm_partition.output_nodes[0]
            layernorm_node = None
            for n in layernorm_partition.nodes:
                if n.target == torch.ops.aten.native_layer_norm.default:
                    layernorm_node = n
            if _is_annotated([output_node, layernorm_node]):  # type: ignore[list-item]
                continue

            input_act = layernorm_node.args[0]  # type: ignore[union-attr]
            assert isinstance(input_act, Node)

            act_qspec = get_input_act_qspec(quantization_config)
            layernorm_node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
                input_qspec_map={
                    input_act: act_qspec,
                },
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )

    def _annotate_add(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        module_partitions = get_source_partitions(
            gm.graph, [operator.add, torch.add]
        )
        add_partitions = list(itertools.chain(*module_partitions.values()))
        for add_partition in add_partitions:
            output_node = add_partition.output_nodes[0]
            add_node = None
            for n in add_partition.nodes:
                if n.target == torch.ops.aten.add.Tensor:
                    add_node = n
            if _is_annotated([output_node, add_node]):  # type: ignore[list-item]
                continue
            
            act_qspec = get_input_act_qspec(quantization_config)
            
            input_qspec_map = {}
            input_qspec_map[add_node.args[0]] = act_qspec
            input_qspec_map[add_node.args[1]] = act_qspec
            
            add_node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
                input_qspec_map=input_qspec_map,
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )
            
    def _annotate_cat(
        self, gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig]
    ) -> None:
        cat_partitions = get_source_partitions(
            gm.graph, [torch.cat]
        )
        cat_partitions = list(itertools.chain(*cat_partitions.values()))
        for cat_partition in cat_partitions:
            cat_node = cat_partition.output_nodes[0]
            if cat_node.target != torch.ops.aten.cat.default:
                # TODO: change this to AnnotationException
                raise Exception(
                    f"Expected cat node: torch.ops.aten.cat.default, but found {cat_node.target}"
                    " please check if you are calling the correct capture API"
                )
                
            if _is_annotated([cat_node]):
                continue

            input_act_qspec = get_input_act_qspec(quantization_config)
            inputs = cat_node.args[0]

            input_qspec_map = {}
            input_act0 = inputs[0]
            if isinstance(input_act0, Node):
                input_qspec_map[input_act0] = input_act_qspec

            for input_act in inputs[1:]:
                input_qspec_map[input_act] = input_act_qspec

            cat_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )
                    
    def validate(self, model: torch.fx.GraphModule) -> None:
        """validate if the annotated graph is supported by the backend"""
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return []