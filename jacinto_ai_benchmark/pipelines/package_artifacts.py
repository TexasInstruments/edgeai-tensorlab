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

import copy
import os
import shutil
import tarfile
import yaml
from . import pipeline_utils
from .. import utils

__all__ = ['package_artifacts']


def package_artifacts(settings, work_dir, pipeline_configs):
    out_dir = work_dir + '_package'
    print(f'packaging artifacts to {out_dir} please wait...')

    for pipeline_id, pipeline_config in pipeline_configs.items():
        package_artifact(pipeline_config, out_dir)
    #


def package_artifact(pipeline_config, package_dir, make_package_dir=True, make_package_tar=True):
    input_files = []
    packaged_files = []

    run_dir = pipeline_config['session'].get_param('run_dir')
    if not os.path.exists(run_dir):
        print(f'could not find: {run_dir}')
        return
    #

    artifacts_folder = pipeline_config['session'].get_param('artifacts_folder')
    if not os.path.exists(artifacts_folder):
        print(f'could not find: {artifacts_folder}')
        return
    #

    # make the top level package_dir
    os.makedirs(package_dir, exist_ok=True)

    # the output run folder
    package_run_dir = os.path.join(package_dir, os.path.basename(run_dir))

    # local model folder
    model_folder = pipeline_config['session'].get_param('model_folder')
    model_path = pipeline_config['session'].get_param('model_path')
    relative_model_dir = os.path.basename(model_folder)
    if isinstance(model_path, (list,tuple)):
        relative_model_path = [os.path.join(relative_model_dir, os.path.basename(m)) for m in model_path]
    else:
        relative_model_path = os.path.join(relative_model_dir, os.path.basename(model_path))
    #

    # local artifacts folder
    artifacts_folder = pipeline_config['session'].get_param('artifacts_folder')
    relative_artifacts_dir = os.path.basename(artifacts_folder)

    # create the param file in source folder with relative paths
    param_file = os.path.join(run_dir, 'param.yaml')
    pipeline_param = pipeline_utils.collect_param(pipeline_config)
    pipeline_param = copy.deepcopy(pipeline_param)
    pipeline_param = utils.pretty_object(pipeline_param)
    pipeline_param['session']['run_dir'] = os.path.basename(run_dir)
    pipeline_param['session']['model_folder'] = relative_model_dir
    pipeline_param['session']['model_path'] = relative_model_path
    pipeline_param['session']['artifacts_folder'] = relative_artifacts_dir
    with open(param_file, 'w') as pfp:
        yaml.safe_dump(pipeline_param, pfp)
    #

    # copy model files
    package_model_folder = os.path.join(package_run_dir, relative_model_dir)
    model_files = utils.list_files(model_folder, basename=False)
    package_model_files = [os.path.join(package_model_folder,os.path.basename(f)) for f in model_files]
    for f, pf in zip(model_files, package_model_files):
        input_files.append(f)
        packaged_files.append(pf)
    #

    # copy artifacts
    package_artifacts_folder = os.path.join(package_run_dir, relative_artifacts_dir)
    artifacts_files = utils.list_files(artifacts_folder, basename=False)
    package_artifacts_files = [os.path.join(package_artifacts_folder,os.path.basename(f)) for f in artifacts_files]
    for f, pf in zip(artifacts_files, package_artifacts_files):
        input_files.append(f)
        packaged_files.append(pf)
    #

    # copy files in run_dir - example result.yaml
    run_files = utils.list_files(run_dir, basename=False)
    package_run_files = [os.path.join(package_run_dir,os.path.basename(f)) for f in run_files]
    for f, pf in zip(run_files, package_run_files):
        input_files.append(f)
        packaged_files.append(pf)
    #

    if make_package_dir:
        for inpf, pf in zip(input_files, packaged_files):
            os.makedirs(os.path.dirname(pf), exist_ok=True)
            shutil.copy2(inpf, pf)
        #
    #

    if make_package_tar:
        tarfile_name = package_run_dir + '.tar.gz'
        tfp = tarfile.open(tarfile_name, 'w:gz')
        for inpf, pf in zip(input_files, packaged_files):
            outpf = pf.replace(package_run_dir, '')
            tfp.add(inpf, arcname=outpf)
        #
        tfp.close()
    #
