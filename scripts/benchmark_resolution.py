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
import sys
import argparse
import functools
import warnings
import copy
import onnx
import warnings
from jai_benchmark import *

try:
    from onnxsim import simplify
except:
    warnings.warn('onnx-simplifier is required to do model import with for this script. '
                  'if you are doing model import, please install it using: '
                  'pip install onnx-simplifier')


# the cwd must be the root of the respository
if os.path.split(os.getcwd())[-1] == 'scripts':
    os.chdir('../')
#

def get_input_shape_onnx(onnx_model, num_inputs=1):
    input_shape = {}
    for input_idx in range(num_inputs):
        input_i = onnx_model.graph.input[input_idx]
        name = input_i.name
        shape = [dim.dim_value for dim in input_i.type.tensor_type.shape.dim]
        input_shape.update({name: shape})
    #
    return input_shape

def get_output_shape_onnx(onnx_model, num_outputs=1):
    output_shape = {}
    num_outputs = 1
    for output_idx in range(num_outputs):
        output_i = onnx_model.graph.output[output_idx]
        name = output_i.name
        shape = [dim.dim_value for dim in output_i.type.tensor_type.shape.dim]
        output_shape.update({name:shape})
    #
    return output_shape


def modify_pipelines(cmds, pipeline_configs_in):
    pipeline_configs_out = {}
    for size_id, input_size in enumerate(cmds.input_sizes):
        # modify a pipeline so that all the models use fixed input size
        # other modifications can also be defined here.
        warning_string = f'Changing input size to {input_size}.\n' \
                         f'The accuracies reported may be wrong as input_size is changed from the default value.'
        print(warning_string)

        for pipeline_id, pipeline_config_in in pipeline_configs_in.items():
            # start with fresh set of configs - not the one modified in earlier iteration
            pipeline_config = copy.deepcopy(pipeline_config_in)
            # start modifying the model for the given input resolution
            preproc_stge = pipeline_config['preprocess']
            preproc_transforms = preproc_stge.transforms
            for tidx, trans in enumerate(preproc_transforms):
                if isinstance(trans, preprocess.ImageResize):
                    trans = preprocess.ImageResize(input_size)
                elif isinstance(trans, preprocess.ImageCenterCrop):
                    trans = preprocess.ImageCenterCrop(input_size)
                #
                preproc_transforms[tidx] = trans
            #
            # generate temporay ONNX  model based on fixed resolution
            model_path = pipeline_config['session'].peek_param('model_path')
            print("=" * 64)
            print("src model path :{}".format(model_path))
            onnx_model = onnx.load(model_path)
            input_name_shapes = get_input_shape_onnx(onnx_model)
            assert len(input_name_shapes) == 1
            input_name = None
            for k, v in input_name_shapes.items():
                input_name = k
            #
            out_name_shapes = get_output_shape_onnx(onnx_model)

            # variable shape model
            input_var_shapes = {input_name: ['b', 3, 'w', 'h']}

            # create a new model_id for the modified model
            new_model_id = pipeline_id[:-1] + str(1+size_id)
            pipeline_config['session'].set_param('model_id', new_model_id)

            # set model_path with the desired size so that the run_dir also will have that size
            # this step is just dummy - the model with the modified name doesn't exist at this point
            model_path_tmp = model_path.replace(".onnx", "_{}x{}.onnx".format(input_size, input_size))
            pipeline_config['session'].set_param('model_path', model_path_tmp)

            # initialize must be called to re-create run_dir and artifacts_folder for this new model_id
            # it is possible that initialize() must have been called before - we need to set run_dir to None to re-create it
            pipeline_config['session'].set_param('run_dir', None)
            pipeline_config['session'].initialize()

            # run_dir must have been created now
            run_dir = pipeline_config['session'].get_param('run_dir')
            model_folder = pipeline_config['session'].get_param('model_folder')

            # now set the final model_path
            model_path_out = os.path.join(model_folder, os.path.basename(model_path_tmp))
            pipeline_config['session'].set_param('model_path', model_path_out)

            # create the modified onnx model with the required input size
            # if the run_dir or the packaged (.tar.gz) artifact is available, this will be skipped
            tarfile_name = run_dir + '.tar.gz'
            linkfile_name = run_dir + '.tar.gz.link'
            if (not os.path.exists(run_dir)) and (not os.path.exists(tarfile_name)) and (not os.path.exists(linkfile_name)):
                # create first varibale shape model
                onnx_model = utils.onnx_update_model_dims(onnx_model, input_var_shapes, out_name_shapes)
                input_name_shapes[input_name] = [1, 3, input_size, input_size]
                # change to fixed shape model
                try:
                    onnx_model, check = simplify(onnx_model, skip_shape_inference=False, input_shapes=input_name_shapes)
                except:
                    warnings.warn(f'changing the size of {model_path} did not work - skipping')
                    continue
                #
                # save model in model_folder
                os.makedirs(model_folder, exist_ok=True)
                print("saving modified model :{}".format(model_path_out))
                onnx.save(onnx_model, model_path_out)
                #onnx.shape_inference.infer_shapes_path(model_path_out, model_path_out)
            #
            pipeline_configs_out.update({new_model_id: pipeline_config})
        #
    #
    return pipeline_configs_out


if __name__ == '__main__':
    print(f'argv={sys.argv}')
    # the cwd must be the root of the respository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #
    model_selection_default = [
                       'edgeai-tv/mobilenet_v1_20190906.onnx',
                       'edgeai-tv/mobilenet_v2_20191224.onnx',
                       'edgeai-tv/mobilenet_v2_1p4_qat-p2_20210112.onnx',
                       'torchvision/resnet18.onnx',
                       'torchvision/resnet50.onnx',
                       'fbr-pycls/regnetx-400mf.onnx',
                       'fbr-pycls/regnetx-800mf.onnx',
                       'fbr-pycls/regnetx-1.6gf.onnx'
                      ]

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('settings_file', type=str, default=None)
    parser.add_argument('--configs_path', type=str)
    parser.add_argument('--models_path', type=str)
    parser.add_argument('--task_selection', type=str, nargs='*')
    parser.add_argument('--model_selection', default=model_selection_default, type=str, nargs='*')
    parser.add_argument('--session_type_dict', type=str, nargs='*')
    parser.add_argument('--input_sizes', default=[512, 1024], type=int, nargs='*')
    cmds = parser.parse_args()

    kwargs = vars(cmds)
    if 'session_type_dict' in kwargs:
        kwargs['session_type_dict'] = utils.str_to_dict(kwargs['session_type_dict'])
    #
    # these artifacts are meant for only performance measurement
    # just do a quick import with simple calibration
    runtime_options = {'accuracy_level': 0}
    settings = config_settings.ConfigSettings(cmds.settings_file,
        num_frames=100, calibration_iterations=1, runtime_options=runtime_options, **kwargs)

    work_dir = os.path.join(settings.modelartifacts_path, f'{settings.tensor_bits}bits')
    print(f'work_dir: {work_dir}')

    # pass a function to modify the pipelines to add the various resolutions
    modify_pipelines_func = functools.partial(modify_pipelines, cmds)

    # run the accuracy pipeline
    tools.run_accuracy(settings, work_dir, modify_pipelines_func=modify_pipelines_func)
