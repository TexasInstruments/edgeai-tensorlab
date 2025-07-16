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
#################################################################################
import os
from . import logger_utils

try:
    from .model_infos import MODEL_INFOS_DICT
except Exception as e:
    MODEL_INFOS_DICT = {}
    print(logger_utils.log_color("\nERROR", "Error in python import of MODEL_INFOS_DICT", str(e)))
    raise

model_id_artifacts_pair = {v['model_id']+'_'+v['session_name']:v['artifact_name']
                           for k, v in MODEL_INFOS_DICT.items()
                           }

shortlisted_model_list = {v['model_id']+'_'+v['session_name']:v['artifact_name']
                          for k, v in MODEL_INFOS_DICT.items()
                          if v['shortlisted']}

recommended_model_list = {v['model_id']+'_'+v['session_name']:v['artifact_name']
                          for k, v in MODEL_INFOS_DICT.items()
                          if v['recommended']}

super_set = list(model_id_artifacts_pair.keys())


def find_compiled_artifact_in_removed_list():
    import yaml
    compiled_artifacts_file = '/data/hdd/a0875091/files/work/bitbucket/model-selection-tool/utils/cloud_eval/perf_acc_rel_8.2_202203/compiled_artifacts.yaml'
    with open(compiled_artifacts_file) as file:
        compiled_artifacts = yaml.full_load(file)
    
        for key, value in compiled_artifacts.items():
            if key not in shortlisted_model_list:
                print("{} for which artifacts are available but model part of removed model list".format(key))


def test_against_super_set():
    for artifacts_id in super_set:
        if not artifacts_id in model_id_artifacts_pair:
            print("{} is part of super-set but not in model names".format(artifacts_id))

def excel_to_dict(excel_file=None, numeric_cols=None):
    try:
        import pandas as pd
    except:
        raise ImportError("excel_to_dict is not supported, check if 'import pandas' work")

    if os.path.splitext(excel_file)[1] == '.xlsx':
        df = pd.read_excel( excel_file, engine='openpyxl')
    elif os.path.splitext(excel_file)[1] == '.csv':
        df = pd.read_csv(excel_file, skipinitialspace=True)
    elif os.path.splitext(excel_file)[1] == '.xls':
        df = pd.read_excel(excel_file)
    else:
        exit(0)

    for pick_key in numeric_cols:
        df[pick_key] = pd.to_numeric(df[pick_key], errors='coerce', downcast='signed').fillna(0.0).astype(float)

    #models_info_list = df.to_dict('list')
    models_info_index = df.to_dict('index')

    #change key form serial number to model_id
    models_info_dict = dict()
    for k,v in models_info_index.items():
        #report changed column name from run_time to runtime_name
        run_time = v['run_time'] if 'run_time' in v else v['runtime_name']
        models_info_dict[v['model_id']+'_'+run_time] = v

    return models_info_dict

def get_missing_models(report_file=None, selected_models_list=None):
    numeric_cols = ['serial_num',	'metric_8bits',	'metric_16bits',	'metric_float',	'metric_reference',	'num_subgraphs',	'infer_time_core_ms',	'ddr_transfer_mb', 'perfsim_ddr_transfer_mb', 'perfsim_gmacs']

    models_info_dict = excel_to_dict(excel_file=report_file, numeric_cols=numeric_cols)

    missing_models_list = [(model, model_id_artifacts_pair[model])for model in selected_models_list if not model in models_info_dict]
    if len(missing_models_list) > 0:
        print("#"*32)
        print("perf-data missing for the models")
        for artifact_id, artifacts_name in sorted(missing_models_list):
            model_id  = artifact_id.split('_')[0]
            #print(model_id,', #', artifact_id, artifacts_name)
            print("'{}', # {:23}, {}".format(model_id, artifact_id, artifacts_name))
        print("#"*32)

    return

def get_selected_models(selected_task=None):
    selected_models_list = [key for key in model_id_artifacts_pair if key in shortlisted_model_list]
    selected_models_for_a_task = [model for model in selected_models_list if model.split('-')[0] == selected_task]
    return selected_models_for_a_task


def get_artifact_name(model_id_or_artifact_id, session_name=None, guess_names=False):
    # artifact_id is model_id followed by session_name
    # either pass a model_id and a session_name
    # or directly pass the artifact_id and don't pass session_name
    if session_name is None:
        artifact_id = model_id_or_artifact_id
    else:
        model_id = model_id_or_artifact_id
        artifact_id = f'{model_id}_{session_name}'
    #

    artifact_name = None
    if artifact_id in model_id_artifacts_pair:
        artifact_name = model_id_artifacts_pair[artifact_id]
    elif guess_names:
        model_id, runtime_name = artifact_id.split('_')
        # create mapping dictionaries
        model_id_to_model_name_dict = {k.split('_')[0]:'-'.join(v.split('-')[1:]) \
                for k,v in model_id_artifacts_pair.items()}
        short_runtime_name_dict = {'tvmdlr':'TVM', 'tflitert':'TFL', 'onnxrt':'ONR'}
        # finally for the artifact name
        if runtime_name in short_runtime_name_dict and model_id in model_id_to_model_name_dict:
            artifact_name = f'{short_runtime_name_dict[runtime_name]}-{model_id_to_model_name_dict[model_id]}'

    return artifact_name


def get_name_key_pair_list(model_ids, session_name, remove_models=True):
    shortlisted_model_list_entries = shortlisted_model_list.keys()
    name_key_pair_list = []
    for model_id in model_ids:
        artifact_id = f'{model_id}_{session_name}'
        artifact_name =  model_id_artifacts_pair[artifact_id] if artifact_id in model_id_artifacts_pair else None
        if artifact_name is not None and \
                (not remove_models or artifact_id in shortlisted_model_list_entries):
            name_key_pair_list.append((artifact_name, model_id))
        #
    #
    return name_key_pair_list


def is_shortlisted_model(artifact_id):
    shortlisted_model_list_entries = shortlisted_model_list.keys()
    is_shortlisted = (artifact_id in shortlisted_model_list_entries)
    return is_shortlisted


def is_recommended_model(artifact_id):
    recommended_model_entries = recommended_model_list.keys()
    is_recommended = (artifact_id in recommended_model_entries)
    return is_recommended


if __name__ == '__main__':
    find_compiled_artifact_in_removed_list()
    generate_list_mising_models = True

    test_against_super_set()

    print("Total models : ", len(model_id_artifacts_pair))
    print("shortlisted models : ", len(shortlisted_model_list))
    print_selected_models = True

    selected_models_list = [key for key in model_id_artifacts_pair if not key not in shortlisted_model_list]

    print("with runtime prefix")
    print("="*64)
    if print_selected_models:
        for selected_model in sorted(selected_models_list):
            print("{}{}{}".format("\'", selected_model,"\'" ), end=',')
        print("")

    print("without runtime prefix")
    print("="*64)
    if print_selected_models:
        for selected_model in sorted(selected_models_list):
            selected_model = '-'.join(selected_model.split('-')[0:-1])
            print("{}{}{}".format("\'", selected_model,"\'" ), end=',')
        print("")
    print("="*64)
    selected_models_vcls = [model for model in selected_models_list if model.split('-')[0] == 'vcls']
    selected_models_vdet = [model for model in selected_models_list if model.split('-')[0] == 'vdet']
    selected_models_vseg = [model for model in selected_models_list if model.split('-')[0] == 'vseg']

    print("num_selected_models: {}, vcls:{}, vdet:{}, vseg:{}".format(len(selected_models_list), len(selected_models_vcls), len(selected_models_vdet), len(selected_models_vseg)))

    #find which models need re-run due to lack of performance data
    if generate_list_mising_models:
        df = get_missing_models(report_file='./work_dirs/modelartifacts/report_20210727-235437.csv',
            selected_models_list=selected_models_list)