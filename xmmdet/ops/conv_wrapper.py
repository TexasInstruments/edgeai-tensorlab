from torch import nn
import mmcv
from mmcv.cnn import CONV_LAYERS
from pytorch_jacinto_ai import xnn

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


@CONV_LAYERS.register_module('ConvDWTriplet')
class ConvDWTriplet2d(nn.Module):
    def __init__(self, *args, with_residual=True, force_residual=False, **kwargs):
        super().__init__()

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

        self.conv = xnn.layers.ConvDWTripletNormAct2d(*args, **kwargs)

        in_planes = args[0]
        out_planes = args[1]
        # kernel_size = kwargs.get('kernel_size', None) or args[2]
        stride = kwargs.get('stride', 1) if len(args)<4 else args[3]
        self.with_residual = force_residual or (with_residual and use_act and (in_planes==out_planes) and (stride==1))
        if self.with_residual:
            self.add = xnn.layers.AddBlock()
        #

    def forward(self, x):
        y = self.conv(x)
        if self.with_residual:
            y = self.add((x,y))
        #
        return y


def ConvModuleWrapper(*args, **kwargs):
    conv_cfg = kwargs.get('conv_cfg', dict(type=None))
    has_type = conv_cfg and ('type' in conv_cfg)

    kernel_size = kwargs.get('kernel_size', None)
    kernel_size = kernel_size or args[2]
    assert kernel_size is not None, 'kernel_size must be specified'

    if not has_type:
        return mmcv.cnn.ConvModule(*args, **kwargs)
    elif conv_cfg.type == 'ConvNormAct' or (conv_cfg.type == 'ConvDWSep' and kernel_size == 1):
        return ConvNormAct2d(*args, **kwargs)
    elif conv_cfg.type == 'ConvDWSep':
        return ConvDWSep2d(*args, **kwargs)
    elif conv_cfg.type == 'ConvDWTriplet':
        return ConvDWTriplet2d(*args, **kwargs)
    else:
        return mmcv.cnn.ConvModule(*args, **kwargs)

