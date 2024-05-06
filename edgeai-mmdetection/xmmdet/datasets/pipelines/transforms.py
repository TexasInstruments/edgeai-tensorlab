from mmdet.datasets.pipelines.transforms import *
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class Bypass(object):
    def __call__(self, results):
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'()'
        return repr_str

