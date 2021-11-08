from torch import nn
import mmcv
from mmcv.cnn import CONV_LAYERS
from torchvision.edgeailite import xnn

################################################################################
# custom conv modules


@CONV_LAYERS.register_module('ConvNormAct')
class ConvNormAct2d(nn.Sequential):
    def __init__(self, *args, **kwargs):
        conv_cfg = kwargs.pop('conv_cfg', dict(type=None))

        kwargs['groups'] = kwargs.get('groups', None) or conv_cfg.get('groups', 1)

        norm_cfg = kwargs.pop('norm_cfg', dict(type='BN'))
        use_norm = norm_cfg is not None and norm_cfg['type'] is not None
        kwargs['normalization'] = conv_cfg.get('normalization', use_norm)

        act_cfg = kwargs.pop('act_cfg', dict(type='ReLU'))
        use_act = act_cfg is not None and act_cfg['type'] is not None
        kwargs['activation'] = conv_cfg.get('activation', use_act)

        use_bias = (not use_norm)
        kwargs['bias'] = kwargs.get('bias', use_bias)

        kwargs.pop('inplace', None)

        blocks = list(xnn.layers.ConvNormAct2d(*args, **kwargs))

        super().__init__(*blocks)


@CONV_LAYERS.register_module('ConvDWSep')
class ConvDWSep2d(nn.Sequential):
    def __init__(self, *args, **kwargs):
        conv_cfg = kwargs.pop('conv_cfg', dict(type=None))

        kwargs['groups'] = kwargs.get('groups', None) or conv_cfg.get('groups', 1)
        kwargs['group_size_dw'] = kwargs.get('group_size_dw', None) or conv_cfg.get('group_size_dw', 1)

        norm_cfg = kwargs.pop('norm_cfg', dict(type='BN'))
        use_norm = norm_cfg is not None and norm_cfg['type'] is not None
        kwargs['normalization'] = conv_cfg.get('normalization', (True, use_norm))

        act_cfg = kwargs.pop('act_cfg', dict(type='ReLU'))
        use_act = act_cfg is not None and act_cfg['type'] is not None
        kwargs['activation'] = conv_cfg.get('activation', (True, use_act))

        use_bias = (not use_norm)
        kwargs['bias'] = kwargs.get('bias', (False, use_bias))

        kwargs.pop('inplace', None)

        blocks = list(xnn.layers.ConvDWSepNormAct2d(*args, **kwargs))

        super().__init__(*blocks)


@CONV_LAYERS.register_module('ConvDWTripletRes')
class ConvDWTripletRes2d(xnn.layers.ConvDWTripletRes2d):
    def __init__(self, *args, **kwargs):
        conv_cfg = kwargs.pop('conv_cfg', dict(type=None))

        kwargs['groups'] = kwargs.get('groups', None) or conv_cfg.get('groups', 1)
        kwargs['group_size_dw'] = kwargs.get('group_size_dw', None) or conv_cfg.get('group_size_dw', 1)

        norm_cfg = kwargs.pop('norm_cfg', dict(type='BN'))
        use_norm = norm_cfg is not None and norm_cfg['type'] is not None
        kwargs['normalization'] = conv_cfg.get('normalization', (True, True, use_norm))

        act_cfg = kwargs.pop('act_cfg', dict(type='ReLU'))
        use_act = act_cfg is not None and act_cfg['type'] is not None
        kwargs['activation'] = conv_cfg.get('activation', (True, True, use_act))

        use_bias = (not use_norm)
        kwargs['bias'] = kwargs.get('bias', (False, False, use_bias))

        kwargs.pop('inplace', None)

        super().__init__(*args, **kwargs)


def ConvModuleWrapper(*args, **kwargs):
    conv_cfg = kwargs.get('conv_cfg', dict(type=None))
    has_type = conv_cfg and ('type' in conv_cfg)

    kernel_size = kwargs.get('kernel_size', None)
    kernel_size = kernel_size or args[2]
    assert kernel_size is not None, 'kernel_size must be specified'

    is_dw_conv = conv_cfg is not None and conv_cfg.type in ('ConvDWSep', 'ConvDWTripletRes', 'ConvDWTripletAlwaysRes')
    if not has_type:
        return mmcv.cnn.ConvModule(*args, **kwargs)
    elif conv_cfg.type == 'ConvNormAct' or (is_dw_conv and kernel_size == 1):
        return ConvNormAct2d(*args, **kwargs)
    elif conv_cfg.type == 'ConvDWSep':
        return ConvDWSep2d(*args, **kwargs)
    elif conv_cfg.type == 'ConvDWTripletRes':
        return ConvDWTripletRes2d(*args, **kwargs)
    elif conv_cfg.type == 'ConvDWTripletAlwaysRes':
        kwargs['always_residual'] = True
        return ConvDWTripletRes2d(*args, **kwargs)
    else:
        return mmcv.cnn.ConvModule(*args, **kwargs)


