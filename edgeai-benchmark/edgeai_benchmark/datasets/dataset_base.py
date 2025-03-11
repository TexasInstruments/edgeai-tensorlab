from .. import utils


class DatasetBase(utils.ParamsBase):
    def __init__(self, **kwargs):
        super().__init__()
        # this is required to save the params
        self.kwargs = kwargs
        # call the utils.ParamsBase.initialize()
        super().initialize()

    def get_color_map(self, num_classes=None):
        if num_classes is None:
            if 'num_classes' in self.kwargs and self.kwargs['num_classes']:
                num_classes = self.kwargs['num_classes']
            else:
                raise RuntimeError("num_classes is not provided")
            #
        #
        color_map = utils.get_color_palette(num_classes)
        return color_map
