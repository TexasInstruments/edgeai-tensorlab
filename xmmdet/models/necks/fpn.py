import warnings
import functools
import copy

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, kaiming_init
from mmcv.cnn import build_activation_layer

from mmdet.core import auto_fp16
from mmdet.models.builder import NECKS
from ...ops import ConvModuleWrapper

from torchvision.edgeailite import xnn


@NECKS.register_module()
class FPNLite(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

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
        super(FPNLite, self).__init__()
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
        self.adds = nn.ModuleList()

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
            self.adds.append(xnn.layers.AddBlock())

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
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
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
            add_block = self.adds[i - 1]
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = add_block((laterals[i - 1], interpolate_fn(laterals[i],**self.upsample_cfg)))
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = add_block((laterals[i - 1], interpolate_fn(laterals[i], size=prev_shape, **self.upsample_cfg)))

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
class BiFPNLite(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, num_outs=None, intermediate_channels=None,
                 add_extra_convs='on_input', extra_convs_on_inputs=True, num_blocks=None, **kwargs):
        super().__init__()
        assert add_extra_convs == 'on_input', 'add_extra_convs must be on_input'
        assert extra_convs_on_inputs == True, 'extra_convs_on_inputs must be True'
        intermediate_channels = out_channels if intermediate_channels is None else intermediate_channels
        self.add_extra_convs = add_extra_convs

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs

        conv_cfg = kwargs.get('conv_cfg', None)
        norm_cfg = kwargs.get('norm_cfg', None)
        self.num_outs_bifpn = num_outs

        blocks = []
        for i in range(num_blocks):
            last_in_channels = [intermediate_channels for _ in range(self.num_outs_bifpn)] if i>0 else in_channels
            if i<(num_blocks-1):
                # the initial bifpn blocks can operate with fewer number of channels
                block_id = i
                bi_fpn = BiFPNLiteBlock(block_id=block_id, in_channels=last_in_channels, out_channels=intermediate_channels,
                                        num_outs=self.num_outs_bifpn, add_extra_convs=add_extra_convs, extra_convs_on_inputs=extra_convs_on_inputs,
                                        **kwargs)
            else:
                # last block can be complex if the intermediate channels are lower - so do up_only			
                block_id = 0 if ((num_blocks == 1) or (out_channels != intermediate_channels)) else i
                up_only = (out_channels != intermediate_channels)
                bi_fpn = BiFPNLiteBlock(block_id=block_id, up_only=up_only, in_channels=last_in_channels, out_channels=out_channels,
                                        num_outs=self.num_outs_bifpn, add_extra_convs=add_extra_convs, extra_convs_on_inputs=extra_convs_on_inputs,
                                        **kwargs)
            #
            blocks.append(bi_fpn)
        #
        self.bifpn_blocks = nn.Sequential(*blocks)

        self.extra_convs = nn.ModuleList()
        if self.num_outs > self.num_outs_bifpn:
            in_ch = self.in_channels[-1] if self.add_extra_convs == 'on_input' else self.out_channels
            DownsampleType = nn.MaxPool2d
            for i in range(self.num_outs-self.num_outs_bifpn):
                extra_conv = build_downsample_module(in_ch, self.out_channels, kernel_size=3, stride=2,
                                                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None,
                                                       DownsampleType=DownsampleType)
                self.extra_convs.append(extra_conv)
                in_ch = self.out_channels
            #
        #

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outputs = self.bifpn_blocks(inputs)
        outputs = list(outputs)
        if self.num_outs > self.num_outs_bifpn:
            inp = inputs[-1] if self.add_extra_convs == 'on_input' else outputs[-1]
            for i in range(self.num_outs-self.num_outs_bifpn):
                extra_inp = self.extra_convs[i](inp)
                outputs.append(extra_inp)
                inp = extra_inp
            #
        #
        return outputs

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, distribution='uniform')


class BiFPNLiteBlock(nn.Module):
    def __init__(self, block_id=None, up_only=False, in_channels=None, out_channels=None, num_outs=5, start_level=0, end_level=-1,
                 add_extra_convs='on_input', extra_convs_on_inputs=True, relu_before_extra_convs=False,
                 no_norm_on_lateral=False, conv_cfg=None, norm_cfg=None, act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(scale_factor=2,mode='bilinear')):
        super(BiFPNLiteBlock, self).__init__()
        assert isinstance(in_channels, list)

        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.up_only = up_only
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
        assert upsample_cfg.scale_factor == 2, 'scale_factor of 2 is recommended'
        assert extra_convs_on_inputs == True, 'extra_convs_on_inputs must be True'
        assert add_extra_convs == 'on_input', 'add_extra_convs must be on_input'
        assert self.relu_before_extra_convs == False, 'this blocks ignores  add_extra_convs'
        assert block_id is not None, f'block_id must be valid: {block_id}'

        # Use act only if conv already has act
        ActType = functools.partial(build_activation_layer, act_cfg) if act_cfg else nn.Identity
        DownsampleType = nn.MaxPool2d
        UpsampleType = xnn.layers.ResizeWith

        # add extra conv layers (e.g., RetinaNet)
        if block_id == 0:
            self.in_channels = []
            self.num_backbone_convs = (self.backbone_end_level - self.start_level)
            self.extra_levels = num_outs - self.num_backbone_convs
            self.in_convs = nn.ModuleList()
            for i in range(num_outs):
                if i < self.num_backbone_convs:
                    in_ch = in_channels[self.start_level + i]
                elif i == self.num_backbone_convs:
                    in_ch = in_channels[-1]
                else:
                    in_ch = out_channels
                #
                self.in_channels.append(in_ch)
                stride = 1 if i < self.num_backbone_convs else 2
                in_conv = build_downsample_module(in_ch, out_channels, kernel_size=3, stride=stride,
                                                            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None,
                                                            DownsampleType=DownsampleType)
                self.in_convs.append(in_conv)
            #
        #

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_acts = nn.ModuleList()
        self.up_adds = nn.ModuleList()
        if not up_only:
            self.downs = nn.ModuleList()
            self.down_convs = nn.ModuleList()
            self.down_acts = nn.ModuleList()
            self.down_adds1 = nn.ModuleList()
            self.down_adds2 = nn.ModuleList()
        #
        for i in range(self.num_outs-1):
            # up modules
            if not up_only:
                up = UpsampleType(**self.upsample_cfg)
                up_conv = ConvModuleWrapper(out_channels,
                        out_channels, 3, padding=1,
                        conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                        act_cfg=None, inplace=False)
                up_act = ActType()
                self.ups.append(up)
                self.up_convs.append(up_conv)
                self.up_acts.append(up_act)
                self.up_adds.append(xnn.layers.AddBlock())
            #
            # down modules
            down = DownsampleType(kernel_size=3, stride=2, padding=1)
            down_conv = ConvModuleWrapper(out_channels,
                out_channels, 3, padding=1,
                conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                act_cfg=None, inplace=False)
            down_act = ActType()
            self.downs.append(down)
            self.down_convs.append(down_conv)
            self.down_acts.append(down_act)
            self.down_adds1.append(xnn.layers.AddBlock())
            self.down_adds2.append(xnn.layers.AddBlock())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        # in convs
        if self.block_id == 0:
            ins = [self.in_convs[i](inputs[self.start_level+i]) for i in range(self.num_backbone_convs)]
            extra_in = inputs[-1]
            for i in range(self.num_backbone_convs, self.num_outs):
                extra_in = self.in_convs[i](extra_in)
                ins.append(extra_in)
            #
        else:
            ins = inputs
        #
        # up convs
        ups = [None] * self.num_outs
        ups[-1] = ins[-1]
        for i in range(self.num_outs-2, -1, -1):
            add_block = self.up_adds[i]
            ups[i] = self.up_convs[i](self.up_acts[i](
                    add_block((ins[i], self.ups[i](ups[i+1])))
            ))
        #
        if self.up_only:
            return tuple(ups)
        else:
            # down convs
            outs = [None] * self.num_outs
            outs[0] = ups[0]
            for i in range(0, self.num_outs-1):
                add_block1 = self.down_adds1[i]
                res = add_block1((ins[i+1], ups[i+1])) if (ins[i+1] is not ups[i+1]) else ins[i+1]
                add_block2 = self.down_adds2[i]
                outs[i+1] = self.down_convs[i](self.down_acts[i](
                    add_block2((res,self.downs[i](outs[i])))
                ))
            #
            return tuple(outs)


def build_downsample_module(in_channels, out_channels, kernel_size, stride,
                                 conv_cfg, norm_cfg, act_cfg, DownsampleType):
    padding = kernel_size//2
    if in_channels == out_channels and stride == 1:
        block = ConvModuleWrapper(in_channels, out_channels, kernel_size=1, stride=1,
                            padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                            act_cfg=act_cfg, inplace=False)
    elif in_channels == out_channels and stride > 1:
        block = DownsampleType(kernel_size=kernel_size, stride=stride, padding=padding)
    elif in_channels != out_channels and stride == 1:
        block = ConvModuleWrapper(in_channels, out_channels, kernel_size=1, stride=stride,
                            padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                            act_cfg=act_cfg, inplace=False)
    else:
        block = nn.Sequential(
                DownsampleType(kernel_size=kernel_size, stride=stride, padding=padding),
                ConvModuleWrapper(in_channels, out_channels, kernel_size=1, stride=1,
                            padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                            act_cfg=act_cfg, inplace=False))
    #
    return block
#