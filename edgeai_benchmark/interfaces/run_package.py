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
import sys
import shutil
import tarfile
import yaml
import glob
import re

from .. import utils


def run_package(settings, work_dir, out_dir, include_results=False, custom_model=False, param_template=None):
    # now write out the package
    package_artifacts(settings, work_dir, out_dir, include_results=include_results, custom_model=custom_model,
                      param_template=param_template)


def match_string(patterns, filename):
    matches = [re.search(p, filename) for p in patterns]
    got_match = any(matches)
    return got_match


def package_artifact(pipeline_param, work_dir, out_dir, make_package_tar=True, make_package_dir=False,
                     include_results=False, param_template=None):
    input_files = []
    packaged_files = []

    run_dir = pipeline_param['session']['run_dir']
    if os.sep not in run_dir:
        run_dir = os.path.join(work_dir, run_dir)
    #
    if not os.path.exists(run_dir):
        print(f'could not find: {run_dir}')
        return None
    #

    artifacts_folder = pipeline_param['session']['artifacts_folder']
    if os.sep not in artifacts_folder:
        artifacts_folder = os.path.join(run_dir, artifacts_folder)
    #
    if not os.path.exists(artifacts_folder):
        print(f'could not find: {artifacts_folder}')
        return None
    #

    # make the top level out_dir
    os.makedirs(out_dir, exist_ok=True)

    # the output run folder
    package_run_dir = os.path.join(out_dir, os.path.basename(run_dir))

    # local model folder
    model_folder = pipeline_param['session']['model_folder']
    if os.sep not in model_folder:
        model_folder = os.path.join(run_dir, model_folder)
    #
    model_path = pipeline_param['session']['model_path']
    relative_model_dir = os.path.basename(model_folder)
    if isinstance(model_path, (list,tuple)):
        relative_model_path = [os.path.join(relative_model_dir, os.path.basename(m)) for m in model_path]
    else:
        relative_model_path = os.path.join(relative_model_dir, os.path.basename(model_path))
    #

    # local artifacts folder
    relative_artifacts_dir = artifacts_folder.replace(run_dir, '')
    relative_artifacts_dir = relative_artifacts_dir.lstrip(os.sep)

    # create the param file in source folder with relative paths
    param_file = os.path.join(run_dir, 'param.yaml')
    pipeline_param = copy.deepcopy(pipeline_param)
    pipeline_param = utils.cleanup_dict(pipeline_param, param_template)

    pipeline_param = utils.pretty_object(pipeline_param)
    pipeline_param['session']['run_dir'] = os.path.basename(run_dir)
    pipeline_param['session']['model_folder'] = relative_model_dir
    pipeline_param['session']['model_path'] = relative_model_path
    pipeline_param['session']['artifacts_folder'] = relative_artifacts_dir
    # result is being excluded from the param.yaml - that will only be in result.yaml
    if 'result' in pipeline_param:
        del pipeline_param['result']
    #
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
    artifacts_patterns = [
        r'_tidl_net.bin$',
        r'_tidl_io_\d*.bin$',
        r'allowedNode.txt$',
        r'_netLog.txt$',
        r'.svg$',
        r'deploy_graph.json$',
        r'deploy_graph.json.*',
        r'deploy_lib.so$',
        r'deploy_lib.so.*',
        r'deploy_params.params$',
        r'deploy_params.params.*',
        # extra files - for information only
        r'netLog.txt$',
        r'layer_info.txt$',
        r'.svg$',
        r'onnxrtMetaData.txt',
        r'dataset.yaml'
    ]
    for f, pf in zip(artifacts_files, package_artifacts_files):
        if match_string(artifacts_patterns, f):
            input_files.append(f)
            packaged_files.append(pf)
        #
    #
    artifacts_folder_tempdir = os.path.join(artifacts_folder, 'tempDir')
    tempfile_patterns = [
        # extra files - for information only
        r'netLog.txt$',
        r'layer_info.txt$',
        r'.svg$']
    if os.path.exists(artifacts_folder_tempdir) and os.path.isdir(artifacts_folder_tempdir):
        tempfiles = utils.list_files(artifacts_folder_tempdir, basename=False)
        package_tempfiles = [os.path.join(package_artifacts_folder,os.path.basename(f)) for f in tempfiles]
        for f, pf in zip(tempfiles, package_tempfiles):
            if match_string(tempfile_patterns, f):
                input_files.append(f)
                packaged_files.append(pf)
            #
        #
    #

    # copy files in run_dir - example result.yaml
    run_files = utils.list_files(run_dir, basename=False)
    package_run_files = [os.path.join(package_run_dir,os.path.basename(f)) for f in run_files]
    for f, pf in zip(run_files, package_run_files):
        if (not include_results) and 'result' in f:
            continue
        #
        input_files.append(f)
        packaged_files.append(pf)
    #

    if make_package_dir:
        for inpf, pf in zip(input_files, packaged_files):
            os.makedirs(os.path.dirname(pf), exist_ok=True)
            shutil.copy2(inpf, pf)
        #
    #

    tarfile_size = 0
    if make_package_tar:
        tarfile_name = package_run_dir + '.tar.gz'
        tfp = tarfile.open(tarfile_name, 'w:gz')
        for inpf, pf in zip(input_files, packaged_files):
            outpf = pf.replace(package_run_dir, '')
            tfp.add(inpf, arcname=outpf)
        #
        tfp.close()
        tarfile_size = os.path.getsize(tarfile_name)
    else:
        package_run_dir = None
    #
    return package_run_dir, tarfile_size


def package_artifacts(settings, work_dir, out_dir, include_results=False, custom_model=False, param_template=None):
    print(f'INFO: packaging artifacts to {out_dir} please wait...')
    run_dirs = glob.glob(f'{work_dir}/*')
    run_dirs = sorted(run_dirs)

    packaged_artifacts_dict = {}
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue
        #
        try:
            param_yaml = os.path.join(run_dir, 'param.yaml')
            result_yaml = os.path.join(run_dir, 'result.yaml')
            read_yaml = result_yaml if os.path.exists(result_yaml) else param_yaml
            with open(read_yaml) as fp:
                pipeline_param = yaml.safe_load(fp)
            #
            package_run_dir, tarfile_size = package_artifact(pipeline_param, work_dir, out_dir,
                                include_results=include_results, param_template=param_template)
            if package_run_dir is not None:
                task_type = pipeline_param['task_type']
                package_run_dir = os.path.basename(package_run_dir)
                model_path = pipeline_param['session']['model_path']
                model_path = model_path[0] if isinstance(model_path, (list,tuple)) else model_path

                run_dir = pipeline_param['session']['run_dir']
                run_dir_basename = os.path.basename(run_dir)
                run_dir_splits = run_dir_basename.split('_')
                artifact_id = '_'.join(run_dir_splits[:2]) if not custom_model else None
                runtime_name = pipeline_param['session']['session_name']
                model_name = utils.get_artifact_name(artifact_id) if not custom_model else None
                model_name = model_name or run_dir_basename

                # artifacts generated using scripts/benchmark_resolution.py will not produce good accuracy
                # that is only for performance test
                suffix_highres = artifact_id.split('_')[0] if not custom_model else ''
                is_highres = suffix_highres.endswith('1') or suffix_highres.endswith('2')
                is_shortlisted = utils.is_shortlisted_model(artifact_id) and (not is_highres) if not custom_model else True
                is_recommended = utils.is_recommended_model(artifact_id) if not custom_model else True

                artifacts_dict = {'task_type': task_type, 'session_name': runtime_name,
                                  'run_dir': package_run_dir, 'model_name': model_name,
                                  'size': tarfile_size,
                                  'shortlisted': is_shortlisted,
                                  'recommended': is_recommended}
                packaged_artifacts_dict.update({artifact_id:artifacts_dict})
                print(utils.log_color('SUCCESS', 'finished packaging', run_dir))
            else:
                print(utils.log_color('WARNING', 'could not package', run_dir))
            #
        except:
            print(utils.log_color('WARNING', 'could not package', run_dir))
        #
        sys.stdout.flush()
    #
    if include_results:
        results_yaml = os.path.join(work_dir, 'results.yaml')
        if os.path.exists(results_yaml):
            package_results_yaml = os.path.join(out_dir, 'results.yaml')
            shutil.copy2(results_yaml, package_results_yaml)
        #
    #
    # sort artifacts
    packaged_artifacts_dict = dict(sorted(packaged_artifacts_dict.items(), key=lambda kv:kv[1]['task_type']))
    # write yaml
    with open(os.path.join(out_dir,'artifacts.yaml'), 'w') as fp:
        yaml.safe_dump(packaged_artifacts_dict, fp)
    #
    # write list
    packaged_artifacts_keys = list(packaged_artifacts_dict.values())[0].keys()
    packaged_artifacts_list = [','.join([str(v) for v in artifact_entry.values()]) \
                               for artifact_entry in packaged_artifacts_dict.values()]
    with open(os.path.join(out_dir,'artifacts.csv'), 'w') as fp:
        fp.write(','.join(packaged_artifacts_keys))
        fp.write('\n')
        fp.write('\n'.join(packaged_artifacts_list))
    #
    with open(os.path.join(out_dir, 'extract.sh'), 'w') as fp:
        # Note: append '-exec rm -f "{}" \;' to delete the original .tar.gz files
        fp.write('find . -name "*.tar.gz" -exec tar --one-top-level -zxvf "{}" \;')
    #


