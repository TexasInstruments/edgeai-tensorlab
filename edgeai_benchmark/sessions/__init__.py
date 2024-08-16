# Copyright (c) 2018-2021, Texas Instruments
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


from .. import constants
from .tvmdlr_session import TVMDLRSession
from .tflitert_session import TFLiteRTSession
from .onnxrt_session import ONNXRTSession


session_name_to_type_dict = {
    constants.SESSION_NAME_TVMDLR : TVMDLRSession,
    constants.SESSION_NAME_TFLITERT: TFLiteRTSession,
    constants.SESSION_NAME_ONNXRT: ONNXRTSession
}


session_type_to_name_dict = {
    TVMDLRSession : constants.SESSION_NAME_TVMDLR,
    TFLiteRTSession : constants.SESSION_NAME_TFLITERT,
    ONNXRTSession: constants.SESSION_NAME_ONNXRT
}


session_name_to_short_name = {
    constants.SESSION_NAME_TVMDLR : 'tvm',
    constants.SESSION_NAME_TFLITERT: 'tfl',
    constants.SESSION_NAME_ONNXRT: 'onr'
}


def get_session_name(session_type):
    assert session_type in session_type_to_name_dict, f'unrecognized session_type: {session_type}'
    return session_type_to_name_dict[session_type]


def get_session_type(session_name):
    assert session_name in session_name_to_type_dict, f'unrecognized session_name: {session_name}'
    return session_name_to_type_dict[session_name]


def get_session_name_to_type_dict():
    return session_name_to_type_dict


# these are some example session configs
# the actual config will vary depending on the parameters used to train the model
def get_common_session_cfg(settings, work_dir=None, input_optimization=True, input_data_layout=constants.NCHW,
                           input_mean=(0.0, 0.0, 0.0), input_scale=(1.0, 1.0, 1.0), **kwargs):
    # if it set to None from outside, change it to True (default value)
    input_optimization = input_optimization if settings.input_optimization is None else settings.input_optimization
    common_session_cfg = dict(work_dir=work_dir, target_machine=settings.target_machine,
              target_device=settings.target_device, run_suffix=settings.run_suffix, tidl_offload=settings.tidl_offload,
              input_optimization=input_optimization, input_data_layout=input_data_layout,
              input_mean=input_mean, input_scale=input_scale,
              run_dir_tree_depth=settings.run_dir_tree_depth,
              **kwargs)
    return common_session_cfg

def get_nomeanscale_session_cfg(settings, work_dir=None, input_optimization=False, input_data_layout=constants.NCHW,
                 input_mean=None, input_scale=None, **kwargs):
    session_cfg = get_common_session_cfg(settings, work_dir, input_optimization=input_optimization,
                input_data_layout=input_data_layout, input_mean=input_mean, input_scale=input_scale, **kwargs)
    return session_cfg

def get_onnx_session_cfg(settings, work_dir=None, input_optimization=True, input_data_layout=constants.NCHW,
                 input_mean=(123.675, 116.28, 103.53), input_scale=(0.017125, 0.017507, 0.017429), **kwargs):
    session_cfg = get_common_session_cfg(settings, work_dir, input_optimization=input_optimization,
                input_data_layout=input_data_layout, input_mean=input_mean, input_scale=input_scale, **kwargs)
    return session_cfg

def get_onnx_bgr_session_cfg(settings, work_dir=None, input_optimization=True, input_data_layout=constants.NCHW,
                 input_mean=(103.53, 116.28, 123.675), input_scale=(0.017429, 0.017507, 0.017125), **kwargs):
    session_cfg = get_common_session_cfg(settings, work_dir, input_data_layout=input_data_layout,
                 input_mean=input_mean, input_scale=input_scale, **kwargs)
    return session_cfg

def get_onnx_quant_session_cfg(settings, work_dir=None, input_optimization=True, **kwargs):
    session_cfg = get_onnx_session_cfg(settings, work_dir, input_optimization=input_optimization, **kwargs)
    return session_cfg

def get_onnx_bgr_quant_session_cfg(settings, work_dir=None, input_optimization=True, **kwargs):
    session_cfg = get_onnx_bgr_session_cfg(settings, work_dir, input_optimization=input_optimization, **kwargs)
    return session_cfg

def get_jai_session_cfg(settings, work_dir=None, input_optimization=True, input_data_layout=constants.NCHW,
                 input_mean=(128.0, 128.0, 128.0), input_scale=(1/64.0, 1/64.0, 1/64.0), **kwargs):
    session_cfg = get_common_session_cfg(settings, work_dir, input_optimization=input_optimization, input_data_layout=input_data_layout,
                 input_mean=input_mean, input_scale=input_scale, **kwargs)
    return session_cfg

def get_jai_quant_session_cfg(settings, work_dir=None, input_optimization=True, **kwargs):
    session_cfg = get_jai_session_cfg(settings, work_dir, input_optimization=input_optimization, **kwargs)
    return session_cfg

def get_mxnet_session_cfg(settings, work_dir=None, input_optimization=True, input_data_layout=constants.NCHW,
                  input_mean=(123.675, 116.28, 103.53), input_scale=(0.017125, 0.017507, 0.017429), **kwargs):
    session_cfg = get_common_session_cfg(settings, work_dir, input_optimization=input_optimization, input_data_layout=input_data_layout,
                 input_mean=input_mean, input_scale=input_scale, **kwargs)
    return session_cfg

def get_tflite_session_cfg(settings, work_dir=None, input_optimization=True, input_data_layout=constants.NHWC,
                 input_mean=(128.0, 128.0, 128.0), input_scale=(1/128.0, 1/128.0, 1/128.0), **kwargs):
    session_cfg = get_common_session_cfg(settings, work_dir, input_optimization=input_optimization, input_data_layout=input_data_layout,
                 input_mean=input_mean, input_scale=input_scale, **kwargs)
    return session_cfg

def get_tflite_quant_session_cfg(settings, work_dir=None, input_optimization=False, input_data_layout=constants.NHWC,
                 input_mean=(0.0, 0.0, 0.0), input_scale=(1.0, 1.0, 1.0), **kwargs):
    session_cfg = get_common_session_cfg(settings, work_dir, input_optimization=input_optimization, input_data_layout=input_data_layout,
                 input_mean=input_mean, input_scale=input_scale, **kwargs)
    return session_cfg
