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
from jacinto_ai_benchmark import *


def main(settings, work_dir):
    ################################################################################################
    # configs for each model pipeline
    # - calibration_dataset: dataset to be used for import - should support __len__ and __getitem__.
    # - input_dataset: dataset to be used for inference - should support __len__ and __getitem__
    #   Output of __getitem__ should be understood by the preprocess stage.
    #   For example, if the dataset returns image filenames, the first entry in the preprocess can be an image read class.
    # - preprocess is just a list of transforms wrapped in utils.TransformsCompose.
    #   It depends on what the dataset class outputs and what the model expects.
    #   We have some default transforms defined in settings.
    # - postprocess is also a list of transforms wrapped in utils.TransformsCompose
    #   It depends on what the model outputs and what the metric evaluation expects.
    # - metric - evaluation metric (eg. accuracy). If metric is not defined in the pipeline,
    #   evaluate() function of the dataset will be called.

    common_cfg = {
        'pipeline_type': settings.pipeline_type,
        'verbose': settings.verbose,
        'run_import': settings.run_import,
        'run_inference': settings.run_inference,
        'calibration_dataset': datasets.ImageNetCls(**settings.imagenet_cls_calib_cfg),
        'input_dataset': datasets.ImageNetCls(**settings.imagenet_cls_val_cfg),
        'postprocess': settings.get_postproc_classification()
    }

    common_session_cfg = dict(work_dir=work_dir, target_device=settings.target_device)
    session_name_to_cfg_dict = settings.get_session_name_to_cfg_dict(is_qat=False)

    pipeline_configs = {
        # torchvision: classification mobilenetv2_224x224 expected_metric: 71.88% top-1 accuracy
        'custom-example1': utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx(),
            session=sessions.TVMDLRSession(**common_session_cfg, **session_name_to_cfg_dict[constants.SESSION_NAME_TVMDLR],
                model_path=f'{settings.modelzoo_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_opset9.onnx')
        ),
        # tensorflow/models: classification mobilenetv1_224x224 expected_metric: 71.0% top-1 accuracy
        'custom-example2': utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite(),
            session=sessions.TFLiteRTSession(**common_session_cfg, **session_name_to_cfg_dict[constants.SESSION_NAME_TFLITERT],
                model_path=f'{settings.modelzoo_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v2_1.0_224.tflite'),
            metric=dict(label_offset_pred=-1)
        ),
        # mxnet : gluoncv model : classification - mobilenetv2_1.0 - accuracy: 72.04% top1
        'custom-example3': utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite(),
            session=sessions.TVMDLRSession(**common_session_cfg, **session_name_to_cfg_dict[constants.SESSION_NAME_TVMDLR],
                model_path=[f'{settings.modelzoo_path}/vision/classification/imagenet1k/gluoncv-mxnet/mobilenetv2_1.0-symbol.json',
                            f'{settings.modelzoo_path}/vision/classification/imagenet1k/gluoncv-mxnet/mobilenetv2_1.0-0000.params']),
        )
    }

    ################################################################################################
    # create runner and run the pipeline
    pipeline_runner = pipelines.PipelineRunner(settings, pipeline_configs)

    # print some info
    run_dirs = [pipeline_config['session'].get_param('run_dir') for model_key, pipeline_config \
                in pipeline_runner.pipeline_configs.items()]
    run_dirs = [os.path.basename(run_dir) for run_dir in run_dirs]
    print(f'configs to run: {run_dirs}')
    print(f'number of configs: {len(pipeline_runner.pipeline_configs)}')

    # now actually run the configs
    if settings.run_import or settings.run_inference:
        pipeline_runner.run()
    #

    # collect the logs and display it
    if settings.collect_results:
        results = pipelines.collect_results(settings, work_dir, pipeline_runner.pipeline_configs, print_results=True)
    #


if __name__ == '__main__':
    # the cwd must be the root of the respository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', type=str)
    cmds = parser.parse_args()
    settings = config_settings.ConfigSettings(cmds.settings_file, model_selection=None)

    expt_name = os.path.splitext(os.path.basename(__file__))[0]
    work_dir = os.path.join('./work_dirs', expt_name, f'{settings.tidl_tensor_bits}bits')
    print(f'work_dir = {work_dir}')

    main(settings, work_dir)
