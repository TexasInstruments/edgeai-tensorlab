from .. import utils


class DatasetBase(utils.ParamsBase):
    def __init__(self, **kwargs):
        super().__init__()
        # this is required to save the params
        self.kwargs = kwargs
        # call the utils.ParamsBase.initialize()
        super().initialize()