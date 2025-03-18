from mmengine.model import is_model_wrapper
from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS

@HOOKS.register_module()
class UseGtDepthHook(Hook):
    def __init__(
        self,
        stop_gt_depth_iter=0,
        stop_iter=0,
    ):
        self.stop_gt_depth_iter = stop_gt_depth_iter
        self.stop_iter = stop_iter

    def before_train_iter(self, runner):
        epoch = runner.epoch
        cur_iter = runner.iter
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        if cur_iter >= self.stop_gt_depth_iter:
            model.pts_bbox_head.flag_disable_gt_depth = True
        if cur_iter >= self.stop_iter:
            model.pts_bbox_head.loss_flag = False

