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
import argparse
import functools
import warnings
import copy
import onnx
from pathlib import Path
from jacinto_ai_benchmark import *

try:
    from onnxsim import simplify
except:
    assert False, 'onnx-simplifier is required for this script. please install using: pip install onnx-simplifier'


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

        # start with fresh set of configs - not the one modified in earlier iteration
        pipeline_configs = copy.deepcopy(pipeline_configs_in)
        for pipeline_id, pipeline_config in pipeline_configs.items():
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
            model_path = pipeline_config['session'].get_param('model_path')
            print("=" * 64)
            print("src model path :{}".format(model_path))
            onnx_model = onnx.load(model_path)
            input_name_shapes = get_input_shape_onnx(onnx_model)
            assert len(input_name_shapes) == 1
            for k, v in input_name_shapes.items():
                input_name = k
            out_name_shapes = get_output_shape_onnx(onnx_model)

            # variable shape model
            input_var_shapes = {input_name: ['b', 3, 'w', 'h']}

            # create first varibale shape model
            onnx_model = utils.onnx_update_model_dims(onnx_model, input_var_shapes, out_name_shapes)
            input_name_shapes[input_name] = [1, 3, input_size, input_size]
            # change to fixed shape model
            onnx_model, check = simplify(onnx_model, skip_shape_inference=True, input_shapes=input_name_shapes)

            # create a new model_id for the modified model
            model_id = pipeline_config['session'].get_param('model_id')
            new_model_id = model_id[:-1] + str(1+size_id)
            pipeline_config['session'].set_param('model_id', new_model_id)

            # initialize must be called to re-create run_dir and artifacts_folder for this new model_id
            pipeline_config['session'].initialize()

            # save model in artifcats path
            artifacts_folder = pipeline_config['session'].get_param('artifacts_folder')
            op_model_path = os.path.join(artifacts_folder, 'tempDir', Path(model_path).name)
            op_model_path = op_model_path.replace(".onnx", "_{}_{}.onnx".format(input_size, input_size))
            pipeline_config['session'].set_param('model_path', op_model_path)

            os.makedirs(os.path.dirname(op_model_path), exist_ok=True)
            print("saving modified model :{}".format(op_model_path))
            onnx.save(onnx_model, op_model_path)
            pipeline_configs_out.update({new_model_id: pipeline_config})
        #
    #
    return pipeline_configs_out


if __name__ == '__main__':
    # the cwd must be the root of the respository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', type=str)
    parser.add_argument('--input_sizes', nargs='*', type=int, default=[512, 1024])
    cmds = parser.parse_args()

    model_selection = [
                       'jai-pytorch/mobilenet_v1_20190906-171544_opset9.onnx',
                       'jai-pytorch/mobilenet_v2_20191224-153212_opset9.onnx',
                       'jai-pytorch/mobilenet_v2_1p4_qat-jai_20210112-093313_opset9.onnx',
                       'torchvision/resnet18_opset9.onnx',
                       'torchvision/resnet50_opset9.onnx',
                       'pycls/RegNetX-400MF_dds_8gpu_opset9.onnx',
                       'pycls/RegNetX-800MF_dds_8gpu_opset9.onnx',
                       'pycls/RegNetX-1.6GF_dds_8gpu_opset9.onnx'
                      ]

    # for performance measurement, we need to use only one frame
    settings = config_settings.ConfigSettings(cmds.settings_file,
            num_frames=1, max_frames_calib=1, max_calib_iterations=1,
            model_selection=model_selection)

    expt_name = os.path.splitext(os.path.basename(__file__))[0]
    work_dir = os.path.join('./work_dirs', expt_name, f'{settings.tidl_tensor_bits}bits')
    print(f'work_dir: {work_dir}')

    if settings.configs_path is not None:
        benchmark_configs = utils.import_folder(settings.configs_path)
        pipeline_configs = benchmark_configs.get_configs(settings, work_dir)
    else:
        pipeline_configs = None
    #

    # pass a function to modify the pipelines to add the various resolutions
    modify_pipelines_func = functools.partial(modify_pipelines, cmds)

    # run the accuracy pipeline
    tools.run_accuracy(settings, work_dir, pipeline_configs, modify_pipelines_func=modify_pipelines_func)
