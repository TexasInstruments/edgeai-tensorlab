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
import cv2
import yaml
import shutil
import PIL
import numpy as np
import onnx
import glob

import edgeai_benchmark

def main(target_device, run_type):

    # the cwd must be the root of the respository
    if os.path.split(os.getcwd())[-1] in ('scripts', 'tutorials'):
        os.chdir('../')
    #

    #########################################################################
    assert ('TIDL_TOOLS_PATH' in os.environ and 'LD_LIBRARY_PATH' in os.environ), \
        "Check the environment variables, TIDL_TOOLS_PATH, LD_LIBRARY_PATH"
    print("TIDL_TOOLS_PATH=", os.environ['TIDL_TOOLS_PATH'])
    print("LD_LIBRARY_PATH=", os.environ['LD_LIBRARY_PATH'])
    print("TARGET_SOC=", target_device)
    print(f"INFO: current dir is: {os.getcwd()}")

    if not os.path.exists(os.environ['TIDL_TOOLS_PATH']):
        print(f"ERROR: TIDL_TOOLS_PATH: {os.environ['TIDL_TOOLS_PATH']} not found")
    else:
        print(f'INFO: TIDL_TOOLS_PATH: {os.listdir(os.environ["TIDL_TOOLS_PATH"])}')
    #

    #########################################################################
    num_frames = 1
    calibrtion_frames = 5 #12
    calibration_iterations = 5 #12
    parallel_processes = 1
    run_type = 'ALL'
    use_dataset_loader = True
    use_config_settings = True

    #########################################################################
    modelartifacts_tempdir_name = os.path.abspath('./work_dirs_custom')
    modelartifacts_custom = os.path.join(modelartifacts_tempdir_name, 'modelartifacts')
    print(f'INFO: clearing modelartifacts folder: {modelartifacts_custom}')
    if (parallel_processes or run_type=='IMPORT') and os.path.exists(modelartifacts_custom):
        shutil.rmtree(modelartifacts_custom, ignore_errors=True)
    #

    #########################################################################
    if use_config_settings:
        settings = edgeai_benchmark.config_settings.CustomConfigSettings('./settings_import_on_pc.yaml',
                        target_device=os.environ['TARGET_SOC'],
                        modelartifacts_path=modelartifacts_custom,
                        model_selection=None, model_shortlist=None,
                        num_frames=100)
    else:
        settings = edgeai_benchmark.core.ConfigRuntimeOptions('./settings_import_on_pc.yaml',
                        target_device=os.environ['TARGET_SOC'],
                        target_machine='pc',
                        modelartifacts_path=modelartifacts_custom,
                        model_selection=None, model_shortlist=None,
                        num_frames=100)
    #


    #########################################################################
    # download dataset if it doesn't exist
    dataset_name='imagenetv2c'
    if not os.path.exists(f'{settings.datasets_path}/{dataset_name}'):
        print(f'INFO: downloading the dataset - {dataset_name}')
        edgeai_benchmark.interfaces.run_download_dataset(settings, dataset_name=dataset_name)
    else:
        print(f'INFO: dataset exists, will reuse - {dataset_name}')
    #

    #########################################################################
    # give the path to your model here
    model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx'
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    work_dir = os.path.join(settings.modelartifacts_path, f'{settings.tensor_bits}bits')
    run_dir = os.path.join(work_dir, os.path.splitext(os.path.basename(model_path))[0])
    shutil.rmtree(run_dir, ignore_errors=True)

    model_file = os.path.join(run_dir, 'model', os.path.basename(model_path))
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    shutil.copy2(model_path, model_file)
    onnx.shape_inference.infer_shapes_path(model_file, model_file)
    print(f'INFO: model_file - {model_file}')

    artifacts_folder = os.path.join(run_dir, 'artifacts')
    os.makedirs(artifacts_folder, exist_ok=True)
    print(f'INFO: artifacts_folder - {artifacts_folder}')


    #########################################################################
    runtime_options = settings.get_runtime_options(
            calibrtion_frames=calibrtion_frames,
            calibration_iterations=calibration_iterations)
    print(f'INFO: runtime_options - {runtime_options}')


    #########################################################################
    def preprocess_input(input_img_file):
        width = 224
        height = 224
        input_mean=[123.675, 116.28, 103.53]
        input_scale=[0.017125, 0.017507, 0.017429]
        input_img = PIL.Image.open(input_img_file).convert("RGB").resize((width, height), PIL.Image.BILINEAR)
        input_data = np.expand_dims(input_img, axis=0)
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        normalized_data = np.zeros(input_data.shape, dtype=np.float32)
        for mean, scale, ch in zip(input_mean, input_scale, range(input_data.shape[1])):
            normalized_data[:, ch, :, :] = (input_data[:, ch, :, :] - mean) * scale
        #
        return normalized_data


    #########################################################################
    def run_import():

        dataset_path = f'{settings.datasets_path}/{dataset_name}'
        images_path = f'{dataset_path}/val'

        if use_dataset_loader:
            dataset_calib_cfg = dict(
                path=images_path,
                split=f'{settings.datasets_path}/{dataset_name}/val.txt',
                num_classes=1000,
                shuffle=True,
                num_frames=10000,
                name='imagenet'
            )
            calib_dataset = edgeai_benchmark.datasets.ImageClassification(**dataset_calib_cfg)
        else:
            calib_dataset = glob.glob(f'{images_path}/*/*.*')
        #

        calib_list = [preprocess_input(calib_dataset[input_index]) for input_index in range(calibrtion_frames)]

        onnxruntime_wrapper = edgeai_benchmark.core.ONNXRuntimeWrapper(
                runtime_options=runtime_options,
                model_file=model_file,
                artifacts_folder=artifacts_folder,
                tidl_tools_path=os.environ['TIDL_TOOLS_PATH'],
                tidl_offload=True)

        onnxruntime_wrapper.run_import(calib_list)
        print(f'INFO: model import done')


    #########################################################################
    def run_inference():
        # dataset parameters for actual inference
        dataset_path = f'{settings.datasets_path}/{dataset_name}'
        images_path = f'{dataset_path}/val'

        if use_dataset_loader:
            dataset_val_cfg = dict(
                path=images_path,
                split=f'{dataset_path}/val.txt',
                num_classes=1000,
                shuffle=True,
                num_frames=min(settings.num_frames,10000),
                name='imagenet'
            )

            val_dataset = edgeai_benchmark.datasets.ImageClassification(**dataset_val_cfg)
        else:
            val_dataset = glob.glob(f'{images_path}/*/*.*')
        #

        onnxruntime_wrapper = edgeai_benchmark.core.ONNXRuntimeWrapper(
                runtime_options=runtime_options,
                model_file=model_file,
                artifacts_folder=artifacts_folder,
                tidl_tools_path=os.environ['TIDL_TOOLS_PATH'],
                tidl_offload=True)

        for input_index in range(num_frames):
            input_data = preprocess_input(val_dataset[input_index])
            outputs = onnxruntime_wrapper.run_inference(input_data)
            print(outputs)

        print(f'INFO: model inference done')


    #########################################################################
    # import and inference can be run in single call if separat3 process is used for them
    # otehrwise one would have to choose between either import or inference in one call of this script.,
    if parallel_processes:
        def run_import_proc():
            proc = edgeai_benchmark.utils.ProcessWithQueue(target=run_import)
            proc.start()
            return proc

        def run_inference_proc():
            proc = edgeai_benchmark.utils.ProcessWithQueue(target=run_inference)
            proc.start()
            return proc

        run_import_task = {'proc_name':f'{model_name}:import','proc_func':run_import_proc}
        run_infernce_task = {'proc_name':f'{model_name}:inference','proc_func':run_inference_proc}
        task_entries = {model_name:[run_import_task, run_infernce_task]}
        parallel_runner = edgeai_benchmark.utils.ParallelRunner(parallel_processes=parallel_processes)
        parallel_runner.run(task_entries)
    elif run_type == "IMPORT":
        run_import()
    elif run_type == "INFERENCE":
        run_inference()
    else:
        assert False, f"ERROR: please set parallel_processes>1 or set run_type to IMPORT or INFERENCE - got {run_type}"


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', type=str, choices=('IMPORT', 'INFERENCE', 'ALL'), default='ALL')
    parser.add_argument('--target_device', type=str, default='AM68A')
    return parser

if __name__ == '__main__':
    print(f'argv: {sys.argv}')
    # the cwd must be the root of the respository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #

    parser = get_arg_parser()
    args = parser.parse_args()

    main(args.target_device, args.run_type)