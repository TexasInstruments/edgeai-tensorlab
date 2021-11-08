import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.fcos_head import FCOSHead

from torchvision.edgeailite import xnn
from ...ops import ConvModuleWrapper

INF = 1e8


@HEADS.register_module()
class FCOSLiteHead(FCOSHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.

    Example:
        >>> self = FCOSLiteHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self, *args, **kwargs):
        self.kernel_size_stack = kwargs.pop('kernel_size_stack', 3)
        self.kernel_size_head = kwargs.pop('kernel_size_head', 3)
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModuleWrapper(
                    chn,
                    self.feat_channels,
                    self.kernel_size_stack,
                    stride=1,
                    padding=(self.kernel_size_stack-1)//2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModuleWrapper(
                    chn,
                    self.feat_channels,
                    self.kernel_size_stack,
                    stride=1,
                    padding=(self.kernel_size_stack-1)//2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.fcos_cls = ConvModuleWrapper(self.feat_channels, self.cls_out_channels, self.kernel_size_head,
                                          padding=(self.kernel_size_head-1)//2,
                                          conv_cfg=self.conv_cfg, norm_cfg=None, act_cfg=None)
        self.fcos_reg = ConvModuleWrapper(self.feat_channels, 4, self.kernel_size_head,
                                          padding=(self.kernel_size_head-1)//2,
                                          conv_cfg=self.conv_cfg, norm_cfg=None, act_cfg=None)
        self.fcos_centerness = ConvModuleWrapper(self.feat_channels, 1, self.kernel_size_head,
                                                 padding=(self.kernel_size_head-1)//2,
                                                 conv_cfg=self.conv_cfg, norm_cfg=None, act_cfg=None)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        # Focal Loss for Dense Object Detection, Tsung-Yi Lin et.al.
        # (RetinaNet paper): https://arxiv.org/pdf/1708.02002.pdf
        # initialization of class bias: sections 3.3, 4.1, 5.1
        # retinanet has only one shared cls head. get the last conv from it
        last_ms = xnn.layers.get_last_bias_modules(self.fcos_cls)
        bias_cls = bias_init_with_prob(0.01)
        if len(last_ms) > 0:
            bias_cls = bias_cls / len(last_ms)
            for m in last_ms:
                normal_init(m, std=0.01, bias=bias_cls)
            #
        #