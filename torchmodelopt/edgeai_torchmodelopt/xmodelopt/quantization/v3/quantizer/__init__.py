
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
# * Neither the quantizer_type of the copyright holder nor the quantizer_types of its
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


class QuantizerTypes:
    XNNPACK = "xnnpack"
    TIDLRT_BASIC = "tidlrt_basic"
    TIDLRT_ADVANCED = "tidlrt_advanced"


def get_quantizer(quantizer_type, annotation_patterns=None, **kwargs):
    if quantizer_type == QuantizerTypes.XNNPACK:
        from .xnnpack import get_quantizer as get_xnnpack_quantizer
        return get_xnnpack_quantizer(annotation_patterns=annotation_patterns, **kwargs)
    elif quantizer_type == QuantizerTypes.TIDLRT_BASIC:
        from .tidlrt.tidlrt_quantizer_basic import get_quantizer as get_tidlrt_basic_quantizer
        return get_tidlrt_basic_quantizer(**kwargs)
    elif quantizer_type == QuantizerTypes.TIDLRT_ADVANCED:
        from .tidlrt.tidlrt_quantizer_advanced import get_quantizer as get_tidlrt_advanced_quantizer
        return get_tidlrt_advanced_quantizer(annotation_patterns=annotation_patterns, **kwargs)
    else:
        raise ValueError(f"ERROR: Quantizer {quantizer_type} not recognized.")


def _unique_list(orig_list):
    '''
    make the elements in the list unqueue, and preserve the order of elements
    Note: list(set(x)) will not preseve the order of elements
    '''
    new_list = []
    for k in orig_list:
        if k not in new_list:
            new_list.append(k)
    return new_list


class QuantizerAnnotationPatterns:
    EXTENDED = ['conv_bn_relu', 'conv_transpose_bn_relu', 'conv_relu', 'conv_transpose_relu', 'conv_bn', 'conv_transpose_bn', 'linear_relu', 'conv', 'linear', 'gru_io_only', 'adaptive_avg_pool2d', 'add_relu', 'mul_relu', 'cat', 'add', 'mul']
    MINIMAL = ['conv_bn_relu', 'conv_transpose_bn_relu', 'conv_relu', 'conv_transpose_relu', 'conv_bn', 'conv_transpose_bn', 'linear_relu', 'conv', 'linear', 'add_relu', 'mul_relu', 'cat', 'add']
    NEW = ['conv_mul_add_relu', 'conv_mul_add', 'mul_add', 'linear_add_relu', 'linear_add', 'matmul']
    FULL = _unique_list(NEW + MINIMAL + EXTENDED)
    DEFAULT = ['conv_bn_relu', 'conv_transpose_bn_relu', 'conv_relu', 'conv_transpose_relu', 'conv_bn', 'conv_transpose_bn', 'linear_relu', 'conv_mul_add_relu', 'conv_mul_add', 'linear_add_relu', 'linear_add', 'conv', 'linear', 'add_relu', 'mul_relu', 'mul_add', 'cat', 'add']
