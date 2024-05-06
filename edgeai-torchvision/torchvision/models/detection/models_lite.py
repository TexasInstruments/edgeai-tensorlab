#################################################################################
# Copyright (c) 2018-2021, Texas Instruments Incorporated - http://www.ti.com
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
#################################################################################

from ...edgeailite import xnn
from . import ssdlite
from . import ssdlite_fpn


__all__ = ['ssdlite_mobilenet_v3_large_lite', 
           'ssdlite_mobilenet_v2_fpn_lite', 'ssdlite_mobilenet_v3_large_fpn_lite', 'ssdlite_mobilenet_v3_small_fpn_lite',
           'ssdlite_regnet_x_400mf_fpn_lite', 'ssdlite_regnet_x_800mf_fpn_lite', 'ssdlite_regnet_x_1_6gf_fpn_lite',
           'ssdlite_efficientnet_b0_fpn_lite', 'ssdlite_efficientnet_b2_fpn_lite',
           'ssdlite_efficientnet_b0_bifpn_lite', 'ssdlite_efficientnet_b2_bifpn_lite'
           ]


# yet to be populated
model_urls = {}


###############################################################################################
# the structure of the mobilenet bacbone itself is changed inside ssdlite.py/SSDLiteFeatureExtractorMobileNet
# so loading the backbone weights is a challenge for this lite model.
def ssdlite_mobilenet_v3_large_lite(**kwargs):
    return xnn.model_surgery.create_lite_model(ssdlite.ssdlite_mobilenet_v3_large,
                pretrained_backbone_names={'^features.':'backbone.features.0.'}, model_urls_dict=model_urls, **kwargs)
				

###############################################################################################
def ssdlite_mobilenet_v2_fpn_lite(**kwargs):
    return xnn.model_surgery.create_lite_model(ssdlite_fpn.ssdlite_mobilenet_v2_fpn,
                pretrained_backbone_names={'^features.':'backbone.body.'}, model_urls_dict=model_urls, **kwargs)


def ssdlite_mobilenet_v3_large_fpn_lite(**kwargs):
    return xnn.model_surgery.create_lite_model(ssdlite_fpn.ssdlite_mobilenet_v3_large_fpn,
                pretrained_backbone_names={'^features.':'backbone.body.'}, model_urls_dict=model_urls, **kwargs)


def ssdlite_mobilenet_v3_small_fpn_lite(**kwargs):
    return xnn.model_surgery.create_lite_model(ssdlite_fpn.ssdlite_mobilenet_v3_small_fpn,
                pretrained_backbone_names={'^features.':'backbone.body.'}, model_urls_dict=model_urls, **kwargs)


###############################################################################################
def ssdlite_regnet_x_400mf_fpn_lite(**kwargs):
    return xnn.model_surgery.create_lite_model(ssdlite_fpn.ssdlite_regnet_x_400mf_fpn,
                pretrained_backbone_names={'^stem.':'backbone.body.stem.', '^trunk_output.':'backbone.body.trunk_output.'},
                group_size_dw=16, model_urls_dict=model_urls, **kwargs)

def ssdlite_regnet_x_800mf_fpn_lite(**kwargs):
    return xnn.model_surgery.create_lite_model(ssdlite_fpn.ssdlite_regnet_x_800mf_fpn,
                pretrained_backbone_names={'^stem.':'backbone.body.stem.', '^trunk_output.':'backbone.body.trunk_output.'},
                group_size_dw=16, model_urls_dict=model_urls, **kwargs)

def ssdlite_regnet_x_1_6gf_fpn_lite(**kwargs):
    return xnn.model_surgery.create_lite_model(ssdlite_fpn.ssdlite_regnet_x_1_6gf_fpn,
                pretrained_backbone_names={'^stem.':'backbone.body.stem.', '^trunk_output.':'backbone.body.trunk_output.'},
                group_size_dw=24, model_urls_dict=model_urls, **kwargs)


###############################################################################################
def ssdlite_efficientnet_b0_fpn_lite(**kwargs):
    return xnn.model_surgery.create_lite_model(ssdlite_fpn.ssdlite_efficientnet_b0_fpn,
                pretrained_backbone_names={'^features.':'backbone.body.'}, model_urls_dict=model_urls, **kwargs)


def ssdlite_efficientnet_b2_fpn_lite(**kwargs):
    return xnn.model_surgery.create_lite_model(ssdlite_fpn.ssdlite_efficientnet_b2_fpn,
                pretrained_backbone_names={'^features.':'backbone.body.'}, model_urls_dict=model_urls, **kwargs)


###############################################################################################
def ssdlite_efficientnet_b0_bifpn_lite(**kwargs):
    return xnn.model_surgery.create_lite_model(ssdlite_fpn.ssdlite_efficientnet_b0_bifpn,
                pretrained_backbone_names={'^features.':'backbone.body.'}, model_urls_dict=model_urls, **kwargs)


def ssdlite_efficientnet_b2_bifpn_lite(**kwargs):
    return xnn.model_surgery.create_lite_model(ssdlite_fpn.ssdlite_efficientnet_b2_bifpn,
                pretrained_backbone_names={'^features.':'backbone.body.'}, model_urls_dict=model_urls, **kwargs)

