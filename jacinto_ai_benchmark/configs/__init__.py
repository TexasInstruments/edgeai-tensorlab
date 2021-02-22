from .. import utils
from . import classification
from . import segmentation
from . import detection


def get_configs(settings, work_dir):
    pipeline_configs = {}
    # merge all the config dictionaries
    for task in utils.as_list(settings.task_selection):
        if task in ('classification', None):
            pipeline_configs.update(classification.get_configs(settings, work_dir))
        #
        if task in ('detection', None):
            pipeline_configs.update(detection.get_configs(settings, work_dir))
        #
        if task in ('segmentation', None):
            pipeline_configs.update(segmentation.get_configs(settings, work_dir))
        #
    #
    return pipeline_configs
