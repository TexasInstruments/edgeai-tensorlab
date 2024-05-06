import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.retina_head import RetinaHead

from torchvision.edgeailite import xnn
from ...ops import ConvModuleWrapper


@HEADS.register_module()
class RetinaLiteHead(RetinaHead):
    """An anchor-based head used in
    `RetinaNet <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaLiteHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
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
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModuleWrapper(
                    chn,
                    self.feat_channels,
                    self.kernel_size_stack,
                    stride=1,
                    padding=(self.kernel_size_stack-1)//2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = ConvModuleWrapper(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            self.kernel_size_head,
            padding=(self.kernel_size_head-1)//2,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        self.retina_reg = ConvModuleWrapper(
            self.feat_channels, self.num_anchors * 4,
            self.kernel_size_head,
            padding=(self.kernel_size_head-1)//2,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        # Focal Loss for Dense Object Detection, Tsung-Yi Lin et.al.
        # (RetinaNet paper): https://arxiv.org/pdf/1708.02002.pdf
        # initialization of class bias: sections 3.3, 4.1, 5.1
        # retinanet has only one shared cls head. get the last conv from it
        last_ms = xnn.layers.get_last_bias_modules(self.retina_cls)
        bias_cls = bias_init_with_prob(0.01)
        if len(last_ms) > 0:
            bias_cls = bias_cls / len(last_ms)
            for m in last_ms:
                normal_init(m, std=0.01, bias=bias_cls)
            #
        #




