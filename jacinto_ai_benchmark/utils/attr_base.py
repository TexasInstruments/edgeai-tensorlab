
class AttrBase:
    def get_param(self, param_name):
        if hasattr(self, param_name):
            return getattr(self, param_name)
        elif param_name in self.kwargs:
            return self.kwargs[param_name]
        else:
            assert False, f'param {param_name} could not be found in object {self.__class__.__name__}'
        #

    def set_param(self, param_name, value):
        if hasattr(self, param_name):
            setattr(self, param_name, value)
        elif param_name in self.kwargs:
            self.kwargs[param_name] = value
        else:
            assert False, f'param {param_name} could not be found in object {self.__class__.__name__}'
        #
