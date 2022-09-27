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

def setup_environment(tidl_tools):
    # these have to be set from the calling environment
    # setting TIDL_BASE_PATH
    # os.environ['TIDL_TOOLS_PATH'] = tidl_tools
    # setting LD_LIBRARY_PATH
    # import_path = f"{tidl_dir}/ti_dl/utils/tidlModelImport/out"
    # rt_path = f"{tidl_dir}/ti_dl/rt/out/PC/x86_64/LINUX/release"
    # tfl_delegate_path = f"{tidl_path}/ti_dl/tfl_delegate/out/PC/x86_64/LINUX/release"
    # ld_library_path = os.environ.get('LD_LIBRARY_PATH','.')
    # ld_library_path = f"{ld_library_path}:{import_path}:{rt_path}:{tfl_delegate_path}"
    # os.environ['LD_LIBRARY_PATH'] = ld_library_path
    # os.environ["TIDL_RT_PERFSTATS"] = "1"
    assert 'TIDL_TOOLS_PATH' in os.environ, 'TIDL_TOOLS_PATH must be set in the calling enviroment'
    assert 'LD_LIBRARY_PATH' in os.environ, 'LD_LIBRARY_PATH must be set in the calling enviroment'
