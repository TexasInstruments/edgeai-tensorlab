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

import os

from .version import __version__
from . import constants
from . import config_settings
from . import pipelines, datasets, preprocess, sessions, postprocess, metrics, utils, interfaces


def get_settings_file(target_machine=constants.TARGET_MACHINE_PC_EMULATION, with_model_import=False):
    supported_machines = (constants.TARGET_MACHINE_PC_EMULATION, constants.TARGET_MACHINE_EVM)
    assert target_machine in supported_machines, f'target_machine must be one of {supported_machines}'
    if target_machine == constants.TARGET_MACHINE_PC_EMULATION or with_model_import:
        settings_file = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../settings_import_on_pc.yaml'))
    elif target_machine == constants.TARGET_MACHINE_EVM:
        settings_file = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../settings_infer_on_evm.yaml'))
    #
    return settings_file
