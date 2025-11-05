
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

from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import OP_TO_ANNOTATOR


if 'OP_TO_ANNOTATOR_BACKUP' not in globals():
    OP_TO_ANNOTATOR_BACKUP = copy.deepcopy(OP_TO_ANNOTATOR)
    STATIC_QAT_ONLY_OPS_BACKUP = copy.deepcopy(XNNPACKQuantizer.STATIC_QAT_ONLY_OPS)
    STATIC_OPS_BACKUP = copy.deepcopy(XNNPACKQuantizer.STATIC_OPS)
    DYNAMIC_OPS_BACKUP = copy.deepcopy(XNNPACKQuantizer.DYNAMIC_OPS)
#


def set_annotation_patterns(annotation_patterns=None):
    # select annotators based on annotation_patterns
    if annotation_patterns:
        OP_TO_ANNOTATOR.clear()
        XNNPACKQuantizer.STATIC_QAT_ONLY_OPS.clear()
        XNNPACKQuantizer.STATIC_OPS.clear()
        XNNPACKQuantizer.DYNAMIC_OPS.clear()
        for n in annotation_patterns:
            if n in OP_TO_ANNOTATOR_BACKUP:
                p = OP_TO_ANNOTATOR_BACKUP[n]
                OP_TO_ANNOTATOR[n] = p
                if n in STATIC_QAT_ONLY_OPS_BACKUP:
                    XNNPACKQuantizer.STATIC_QAT_ONLY_OPS +=[n]
                #
                if n in STATIC_OPS_BACKUP:
                    XNNPACKQuantizer.STATIC_OPS +=[n]
                #
                if n in DYNAMIC_OPS_BACKUP:
                    XNNPACKQuantizer.DYNAMIC_OPS +=[n]
                #
            else:
                print(f"WARNING: Annotation pattern {n} not not one of: {OP_TO_ANNOTATOR_BACKUP.keys()}")
            #
        #
    #


def get_quantizer(device=None, annotation_patterns=None, **kwargs):
    set_annotation_patterns(annotation_patterns=annotation_patterns)
    return XNNPACKQuantizer()
