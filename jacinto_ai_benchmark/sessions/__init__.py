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


def get_session_name(session_type):
    assert session_type in session_type_to_name_dict, f'unrecognized session_type: {session_type}'
    return session_type_to_name_dict[session_type]


def get_session_type(session_name):
    assert session_name in session_name_to_type_dict, f'unrecognized session_name: {session_name}'
    return session_name_to_type_dict[session_name]


def get_session_name_to_type_dict():
    return session_name_to_type_dict

