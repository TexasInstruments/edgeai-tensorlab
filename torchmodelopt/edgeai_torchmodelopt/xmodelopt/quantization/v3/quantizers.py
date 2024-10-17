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
import torch.ao.quantization
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
    QuantizationSpec,
    DerivedQuantizationSpec
)
import itertools
import operator
from typing import Dict, List, Optional, Any

from torch.fx import Node

from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from . import qconfig_types

import warnings

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
    if node.target in (torch.ops.aten.softmax.int, torch.ops.aten.relu.default, torch.ops.aten.sigmoid.default) :
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
    elif node.target == torch.ops.aten.linear.default:
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

def _derive_bias_qparams_fn(
        obs_or_fqs: List,
    ):
        assert (
            len(obs_or_fqs) == 2
        ), f"Expecting two obs/fqs, one for activation and one for weight, got: {len(obs_or_fqs)}"
        act_obs_or_fq = obs_or_fqs[0]
        weight_obs_or_fq = obs_or_fqs[1]
        weight_scale, weight_zp = weight_obs_or_fq.calculate_qparams()
        act_scale, act_zp = act_obs_or_fq.calculate_qparams()
        (broadcast_act_scale, broadcast_weight_scale) = torch.broadcast_tensors(
            act_scale, weight_scale
        )
        derived_scale = (broadcast_act_scale * broadcast_weight_scale).to(torch.float32)
        derived_zero = torch.zeros(derived_scale.size()).to(torch.int32)
        return (derived_scale, derived_zero)

def _derived_bias_quant_spec(weight_node, input_act_node, curr_node) -> DerivedQuantizationSpec:

    if curr_node is not None:
        return DerivedQuantizationSpec(
            derived_from=[(input_act_node, curr_node), (weight_node, curr_node)],
            derive_qparams_fn=_derive_bias_qparams_fn,
            dtype=torch.int32,
            quant_min=torch.iinfo(torch.int32).min,
            quant_max=torch.iinfo(torch.int32).max,
            ch_axis=0,
            qscheme=torch.per_channel_symmetric,
        )
    else:
        return DerivedQuantizationSpec(
            derived_from=[input_act_node, weight_node],
            derive_qparams_fn=_derive_bias_qparams_fn,
            dtype=torch.int32,
            quant_min=torch.iinfo(torch.int32).min,
            quant_max=torch.iinfo(torch.int32).max,
            ch_axis=0,
            qscheme=torch.per_channel_symmetric,
        )

class TIDLRTQuantizer(Quantizer):

    def __init__(self, is_qat, fast_mode=False, is_fake_quantize=True):
        super().__init__()
        self.global_config: QuantizationConfig = None  # type: ignore[assignment]
        self.operator_type_config: Dict[str, Optional[QuantizationConfig]] = {}
        self.is_qat = is_qat 
        self.fast_mode = fast_mode
        self.is_fake_quantize = is_fake_quantize
        self.single_input_single_output_shared_nodes = [torch.ops.aten.max_pool2d.default, 
                                                        torch.ops.aten.flatten.using_ints, 
                                                        torch.ops.aten.slice.Tensor,
                                                        torch.ops.aten.dropout.default,
                                                        torch.ops.aten.reshape.default]
        self.single_input_single_output_different_nodes = [ torch.ops.aten.leaky_relu.default, 
                                                            torch.ops.aten.gelu.default,
                                                            torch.ops.aten.relu.default,
                                                            torch.ops.aten.softmax.int,
                                                            torch.ops.aten.mul.Tensor, 
                                                            torch.ops.aten.div.Tensor]

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
        self._annotate_single_input_single_output_different(model, config)
        self._annotate_single_input_single_output_shared(model, config)
        self._annotate_layernorm(model, config) # the weights and bias also need to be quantized
        self._annotate_cat(model, config)
        self._annotate_add(model, config)
        self._annotate_view(model, config) # view possible in loss which creates issues, mostly it needs to be quantized to support bias quantization.
        self._annotate_matmul(model, config)
        self._annotate_conv2d(model, config, allow_16bit_node_list)
        self._annotate_linear(model, config, allow_16bit_node_list)
        return model

    def _annotate_single_input_single_output_shared(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        for node in gm.graph.nodes:
            # skip annotation if it is already annotated
            if _is_annotated([node]):
                continue
            if node.target in self.single_input_single_output_shared_nodes:
                input_act = node.args[0]
                node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
                input_qspec_map={
                    input_act: get_input_act_qspec(quantization_config),
                },
                output_qspec=SharedQuantizationSpec((input_act, node)),
                _annotated=True,
            )
                
    def _annotate_single_input_single_output_different(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        for node in gm.graph.nodes:
            # skip annotation if it is already annotated
            if _is_annotated([node]):
                continue
            if node.target in self.single_input_single_output_different_nodes:
                input_act = node.args[0]
                node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
                input_qspec_map={
                    input_act: get_input_act_qspec(quantization_config),
                },
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )

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
                or conv_node.target != torch.ops.aten.conv2d.default
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

            if len(conv_node.args) >= 3 and (bias := conv_node.args[2]) is not None:
                if isinstance(bias, Node):
                    input_qspec_map[bias] = _derived_bias_quant_spec(weight, input_act, conv_node)

            conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
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
                    if _is_annotated([weight_node]) is False:  # type: ignore[list-item]
                        _annotate_output_qspec(weight_node, weight_qspec)
                    if _is_annotated([output_node]) is False:
                        _annotate_output_qspec(output_node, act_qspec)
                    if bias_node and _is_annotated([bias_node]) is False:
                        if _is_annotated([act_node]):
                            bias_qspec = _derived_bias_quant_spec(weight_node, act_node, None)
                        else:
                            warnings.warn("The bias for the node {} would not be quantized, it might not give correct results !".format(act_use_node.name))
                            bias_qspec = get_bias_qspec(quantization_config)
                        _annotate_output_qspec(bias_node, bias_qspec)
                    nodes_to_mark_annotated = list(p.nodes)
                    _mark_nodes_as_annotated(nodes_to_mark_annotated)
            
    def _annotate_view(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        for node in gm.graph.nodes:
            if node.target == torch.ops.aten.view.default and next(iter(node.users)).target == torch.ops.aten.linear.default:
                # the nodes in loss are also getting quantized which is not desired
                # TODO ignore the loss part of the network for the quantizer aspect
                input_act = node.args[0]
                node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
                input_qspec_map={
                    input_act: get_input_act_qspec(quantization_config),
                },
                output_qspec=SharedQuantizationSpec((input_act, node)),
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
                if n.target in (torch.ops.aten.matmul.default, torch.ops.aten.bmm.default):
                    matmul_node = n
            if _is_annotated([output_node, matmul_node]):  # type: ignore[list-item]
                continue
            
            act_qspec = get_input_act_qspec(quantization_config)
            # setting the symmetric inputs
            # messy way of doing it, need better #TODO
            if hasattr(act_qspec.observer_or_fake_quant_ctr, "p") and hasattr(act_qspec.observer_or_fake_quant_ctr.p, "keywords"):
                observer = act_qspec.observer_or_fake_quant_ctr.p.keywords['observer']
            else:
                observer = act_qspec.observer_or_fake_quant_ctr
            act_qspec_symmetric = qconfig_types.get_act_quantization_config(
                dict(
                    qscheme=torch.per_tensor_symmetric, 
                    power2_scale=observer.__init__._partialmethod.keywords['power2_scale'], 
                    range_shrink_percentile=observer.__init__._partialmethod.keywords['range_shrink_percentile']
                ),
                is_fake_quantize=self.is_fake_quantize,
                fast_mode=self.fast_mode
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
            input_node = layernorm_partition.input_nodes[0]
            div_node = [node for node in layernorm_partition.nodes if node.target==torch.ops.aten.div.Tensor][0]
            mul_node = [node for node in layernorm_partition.nodes if node.target==torch.ops.aten.mul.Tensor][0]
            add_node = output_node
            
            mul_inp = [node for node in mul_node.args if node.op == 'get_attr'][0]
            add_inp = [node for node in add_node.args if node.op == 'get_attr'][0]
            
            weight_qspec = get_weight_qspec(quantization_config)
            act_qspec = get_input_act_qspec(quantization_config)
            
            _annotate_output_qspec(input_node, act_qspec)
            
            div_node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )
            
            mul_node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
                input_qspec_map={
                    mul_inp: act_qspec,
                },
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )
            
            add_node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
                input_qspec_map={
                    add_inp: act_qspec,
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