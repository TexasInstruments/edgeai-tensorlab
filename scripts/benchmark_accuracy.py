import os
import argparse
from jacinto_ai_benchmark import *

# the cwd must be the root of the respository
if os.path.split(os.getcwd())[-1] == 'scripts':
    os.chdir('../')
#

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', type=str)
    cmds = parser.parse_args()
    settings = config_settings.ConfigSettings(cmds.settings_file)

    expt_name = os.path.splitext(os.path.basename(__file__))[0]
    work_dir = os.path.join('./work_dirs', expt_name, f'{settings.tidl_tensor_bits}bits')
    print(f'work_dir = {work_dir}')

    pipeline_configs = {}
    # merge all the config dictionaries
    task_selection = utils.as_list(settings.task_selection)
    for task in task_selection:
        if settings.task_selection in ('classification', None):
            pipeline_configs.update(configs.classification.get_configs(settings, work_dir))
        #
        if settings.task_selection in ('detection', None):
            pipeline_configs.update(configs.detection.get_configs(settings, work_dir))
        #
        if settings.task_selection in ('segmentation', None):
            pipeline_configs.update(configs.segmentation.get_configs(settings, work_dir))
        #
    #

    # create the pipeline_runner which will manage the sessions.
    pipeline_runner = pipelines.PipelineRunner(settings, pipeline_configs)

    # at this point, pipeline_runner.pipeline_configs is a dictionary that has the selected configs
    # some examples of accessing param from it - here 0th entry is used an example.
    # pipeline_config = pipeline_runner.pipeline_configs.values()[0]
    # pipeline_config['preprocess'].get_param('resize') gives the resize dimension
    # pipeline_config['preprocess'].get_param('crop') gives the crop dimension
    # pipeline_config['session'].get_param('run_dir') gives the folder where artifacts are located

    # now actually run the configs
    if settings.run_import or settings.run_inference:
        pipeline_runner.run()
    #

    # collect the logs and display it
    if settings.collect_results:
        results = pipelines.collect_results(settings, work_dir)
        print(*results, sep='\n')
    #
