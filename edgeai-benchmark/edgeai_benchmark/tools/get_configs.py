from .. import utils
from .. import datasets


def get_configs(settings, work_dir):
    # import the configs module
    configs_module = utils.import_folder(settings.configs_path)
    pipeline_configs = configs_module.get_configs(settings, work_dir)
    return pipeline_configs


def select_configs(settings, work_dir, session_name=None):
    # import the configs module
    configs_module = utils.import_folder(settings.configs_path)
    pipeline_configs = configs_module.select_configs(settings, work_dir, session_name)
    return pipeline_configs
