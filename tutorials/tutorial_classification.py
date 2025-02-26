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
import cv2
import yaml
import shutil
from edgeai_benchmark import *

# the cwd must be the root of the respository
if os.path.split(os.getcwd())[-1] in ('scripts', 'tutorials'):
    os.chdir('../')
#

assert ('TIDL_TOOLS_PATH' in os.environ and 'LD_LIBRARY_PATH' in os.environ and
        'TARGET_SOC' in os.environ), "Check the environment variables"
print("TIDL_TOOLS_PATH=", os.environ['TIDL_TOOLS_PATH'])
print("LD_LIBRARY_PATH=", os.environ['LD_LIBRARY_PATH'])
print("TARGET_SOC=", os.environ['TARGET_SOC'])

print(f"INFO: current dir is: {os.getcwd()}")


modelartifacts_tempdir_name = os.path.abspath('./work_dirs_custom')
modelartifacts_custom = os.path.join(modelartifacts_tempdir_name, 'modelartifacts')
print(f'INFO: clearing modelartifacts: {modelartifacts_custom}')
if os.path.exists(modelartifacts_custom):
    shutil.rmtree(modelartifacts_custom)
#

settings = config_settings.CustomConfigSettings('./settings_import_on_pc.yaml',
                target_device = os.environ['TARGET_SOC'],
                modelartifacts_path=modelartifacts_custom,
                num_frames=100)

work_dir = os.path.join(settings.modelartifacts_path, f'{settings.tensor_bits}bits')



dataset_calib_cfg = dict(
    path=f'{settings.datasets_path}/imagenetv2c/val',
    split=f'{settings.datasets_path}/imagenetv2c/val.txt',
    num_classes=1000,
    shuffle=True,
    num_frames=10000,
    name='imagenet'
)

# dataset parameters for actual inference
dataset_val_cfg = dict(
    path=f'{settings.datasets_path}/imagenetv2c/val',
    split=f'{settings.datasets_path}/imagenetv2c/val.txt',
    num_classes=1000,
    shuffle=True,
    num_frames=min(settings.num_frames,10000),
    name='imagenet'
)

calib_dataset = datasets.ImageClassification(**dataset_calib_cfg)
val_dataset = datasets.ImageClassification(**dataset_val_cfg)


# choose one session_name depending on the model type
# tflitert for tflite models, onnxrt for onnx model
#session_name = constants.SESSION_NAME_TFLITERT
session_name = constants.SESSION_NAME_ONNXRT

session_type = settings.get_session_type(session_name)
runtime_options = settings.get_runtime_options(session_name, is_qat=False)

print(session_type)
print(runtime_options)


preproc_transforms = preprocess.PreProcessTransforms(settings)
postproc_transforms = postprocess.PostProcessTransforms(settings)

# these session cfgs also has some default input mean and scale.
# if your model needs a difference mean and scale, update the session cfg dict being used with those values
onnx_session_cfg = sessions.get_onnx_session_cfg(settings, work_dir=work_dir)
tflite_session_cfg = sessions.get_tflite_session_cfg(settings, work_dir=work_dir)


pipeline_configs = {
    'cl-mnv2': dict(
        task_type='classification',
        calibration_dataset=calib_dataset,
        input_dataset=val_dataset,
        preprocess=preproc_transforms.get_transform_onnx(),
        session=session_type(**onnx_session_cfg,
            runtime_options=runtime_options,
            model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx'),
        postprocess=postproc_transforms.get_transform_classification(),
        model_info=dict(metric_reference={'accuracy_top1%':71.88})
    ),
}
print(f"\nINFO: pipeline_configs={pipeline_configs}")


# run the model compliation/import and inference
interfaces.run_benchmark_config(settings, work_dir, pipeline_configs)
run_dir = list(pipeline_configs.values())[0]['session'].get_run_dir()
print(f"\nINFO: compiled artifacts is in: {run_dir}")

# print the result:
with open(os.path.join(run_dir, "result.yaml")) as fp:
    result_yaml = yaml.safe_load(fp)
    result_dict = result_yaml['result']
    print(f"INFO: result - {result_dict}")
#


print()
out_dir = f'{work_dir}_package'
interfaces.package_artifacts(settings, work_dir, out_dir)
print(f'\nINFO: download the atricats files from the folder: {out_dir}')
print(os.listdir(out_dir))

