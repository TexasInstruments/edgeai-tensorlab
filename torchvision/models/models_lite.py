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

from ..edgeailite import xnn
from . import mobilenetv2
from . import mobilenetv3
from . import efficientnet


__all__ = ['mobilenet_v2_lite',
           'mobilenet_v3_large_lite', 'mobilenet_v3_small_lite',
           'efficientnet_b0_lite', 'efficientnet_b1_lite', 'efficientnet_b2_lite',
           'efficientnet_b3_lite', 'efficientnet_b4_lite'
           ]

model_url_base = \
    'https://software-dl.ti.com/jacinto7/esd/modelzoo/stable/models/vision/classification/imagenet1k/edgeai-tv'


model_urls = {
'mobilenet_v2_lite': f'{model_url_base}/mobilenet_v2_20191224_checkpoint.pth',
'mobilenet_v3_large_lite': f'{model_url_base}/mobilenet_v3_lite_large_20210507_checkpoint.pth',
'mobilenet_v3_small_lite': f'{model_url_base}/mobilenet_v3_lite_small_20210429_checkpoint.pth',
}

###############################################################################################
def mobilenet_v2_lite(**kwargs):
    return xnn.model_surgery.create_lite_model(mobilenetv2.mobilenet_v2, model_urls_dict=model_urls, **kwargs)


###############################################################################################
def mobilenet_v3_large_lite(**kwargs):
    return xnn.model_surgery.create_lite_model(mobilenetv3.mobilenet_v3_large, model_urls_dict=model_urls, **kwargs)


def mobilenet_v3_small_lite(**kwargs):
    return xnn.model_surgery.create_lite_model(mobilenetv3.mobilenet_v3_small, model_urls_dict=model_urls, **kwargs)


###############################################################################################
def efficientnet_b0_lite(pretrained=None, **kwargs):
    return xnn.model_surgery.create_lite_model(efficientnet.efficientnet_b0, model_urls_dict=model_urls, **kwargs)


def efficientnet_b1_lite(pretrained=None, **kwargs):
    return xnn.model_surgery.create_lite_model(efficientnet.efficientnet_b1, model_urls_dict=model_urls, **kwargs)


def efficientnet_b2_lite(pretrained=None, **kwargs):
    return xnn.model_surgery.create_lite_model(efficientnet.efficientnet_b2, model_urls_dict=model_urls, **kwargs)


def efficientnet_b3_lite(pretrained=None, **kwargs):
    return xnn.model_surgery.create_lite_model(efficientnet.efficientnet_b3, model_urls_dict=model_urls, **kwargs)


def efficientnet_b4_lite(pretrained=None, **kwargs):
    return xnn.model_surgery.create_lite_model(efficientnet.efficientnet_b4, model_urls_dict=model_urls, **kwargs)
