import warnings
import functools

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.cnn import build_activation_layer

from mmdet.core import auto_fp16
from mmdet.models.builder import NECKS
from ...ops import ConvModuleWrapper

from pytorch_jacinto_ai import xnn


@NECKS.register_module()
class JaiFPN(nn.Module):
    """
    Feature Pyramid Network.

    This is an implementation of - Feature Pyramid Networks for Object
    Detection (https://arxiv.org/abs/1612.03144)

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(JaiFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModuleWrapper(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModuleWrapper(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModuleWrapper(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # using resize_with instead of F.interpolate to generate compact onnx graph
        interpolate_fn = xnn.layers.resize_with

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += interpolate_fn(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += interpolate_fn(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


@NECKS.register_module()
class JaiBiFPN(nn.Sequential):
    def __init__(self, *args, num_blocks=None, **kwargs):
        blocks = []
        for i in range(num_blocks):
            bi_fpn = JaiBiFPNBlock(*args, block_id=i, **kwargs)
            blocks.append(bi_fpn)
        #
        super().__init__(*blocks)


class JaiBiFPNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs='on_input',
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(scale_factor=2,mode='nearest'),
                 block_id=None):
        super(JaiBiFPNBlock, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.block_id = block_id

        assert upsample_cfg is not None, 'upsample_cfg must not be None'
        assert upsample_cfg.mode == 'nearest', 'nearest upsample is recommended'
        assert upsample_cfg.scale_factor == 2, 'scale_factor of 2 is recommended'
        assert extra_convs_on_inputs == True,  \
            'this version of FPN supports only extra_convs_on_inputs == True'
        assert add_extra_convs == 'on_input', \
            'this version of FPN supports only add_extra_convs == on_input'
        assert self.relu_before_extra_convs == False, \
            'this version of FPN supports only relu_before_extra_convs == False'
        assert block_id is not None, f'block_id must be valid: {block_id}'
        if act_cfg is None:
            warnings.warn(f'better to use act_cfg and set activation for this class: {self.__class__.__name__}')
        #

        # Use act only if conv already has act
        ActType = functools.partial(build_activation_layer, act_cfg) if act_cfg else nn.Identity
        DownType = nn.MaxPool2d

        # add extra conv layers (e.g., RetinaNet)
        if block_id == 0:
            self.num_backbone_convs = (self.backbone_end_level - self.start_level)
            self.extra_levels = num_outs - self.num_backbone_convs
            self.in_convs = nn.ModuleList()
            for i in range(num_outs):
                if i < self.num_backbone_convs:
                    in_conv = ConvModuleWrapper(
                        in_channels[self.start_level + i],
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False)
                elif i == (num_outs - self.extra_levels):
                    in_conv = nn.Sequential(
                        DownType(
                            kernel_size=3,
                            stride=2,
                            padding=1),
                        ConvModuleWrapper(
                        in_channels[self.start_level + i],
                        out_channels,
                        kernel_size=1,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False))
                else:
                    in_conv = DownType(
                        kernel_size=3,
                        stride=2,
                        padding=1)
                #
                self.in_convs.append(in_conv)

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_acts = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        self.down_acts = nn.ModuleList()
        for i in range(self.num_outs-1):
            # up modules
            up = nn.Upsample(
                    **self.upsample_cfg
                    )
            up_conv = ConvModuleWrapper(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
            up_act = ActType()
            self.ups.append(up)
            self.up_convs.append(up_conv)
            self.up_acts.append(up_act)
            # down modules
            down = DownType(kernel_size=3, stride=2, padding=1)
            down_conv = ConvModuleWrapper(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            down_act = ActType()
            self.downs.append(down)
            self.down_convs.append(down_conv)
            self.down_acts.append(down_act)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        # in convs
        if self.block_id > 0:
            ins = inputs
        else:
            ins = [self.in_convs[i](inputs[i]) for i in range(self.num_backbone_convs)]
            extra_in = inputs[-1]
            for i in range(self.num_backbone_convs, self.num_outs):
                extra_in = self.in_convs[i](extra_in)
                ins.append(extra_in)
            #
        #
        # up convs
        ups = [None] * self.num_outs
        ups[-1] = ins[-1]
        for i in range(self.num_outs-2, 0, -1):
            ups[i] = self.up_convs[i](self.up_acts(ins[i] + self.ups[i](ups[i+1])))
        #
        # down convs
        outs = [None] * self.num_outs
        outs[0] = ups[0]
        for i in range(0, self.num_outs-1):
            outs[i+1] = self.down_convs[i](self.down_acts(ins[i+1] + ups[i+1] + self.downs[i](ups[i])))
        #
        return tuple(outs)
