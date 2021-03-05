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
import glob
import pickle
from .. import utils


def collect_results(settings, work_dir, print_results=True):
    results = []
    logs_dirs = glob.glob(f'{work_dir}/*')
    for log_dir in logs_dirs:
        if os.path.isdir(log_dir):
            pkl_filename = f'{log_dir}/result.pkl'
            try:
                with open(pkl_filename, 'rb') as pkl_fp:
                    result = pickle.load(pkl_fp)
                    result = correct_result(result)
                    results.append(result)
                #
            except:
                pass
            #
        #
    #
    results = sorted(results, key=lambda item: item['infer_path'])
    if settings.enable_logging:
        result_filename = os.path.join(work_dir, 'results.log')
        with open(result_filename,'w') as writer_fp:
            for result in results:
                writer_fp.write(f'\n{utils.round_dict(result)}')
            #
        #
        pkl_filename = os.path.join(work_dir, 'results.pkl')
        with open(pkl_filename, 'wb') as fp:
            pickle.dump(results, fp)
        #
    #
    if print_results:
        for result in results:
            print(utils.round_dict(result))
        #
    #
    return results


def correct_result(result):
    if 'inference_path' in result:
        result['infer_path'] = result['inference_path']
        del result['inference_path']
    #
    return result