# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector


@MODELS.register_module()
class YOLOX(SingleStageDetector):
    r"""Implementation of `YOLOX: Exceeding YOLO Series in 2021
    <https://arxiv.org/abs/2107.08430>`_

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLOX. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLOX. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    def surgery_init(self, surgery_wrapper, **kwargs):
        self.backbone = surgery_wrapper(self.backbone, **kwargs)
        self.neck = surgery_wrapper(self.neck, **kwargs)
        # handle self.bbox_head
        num_heads = len(self.bbox_head.multi_level_cls_convs)
        for head_index in range(num_heads):
            self.bbox_head.multi_level_cls_convs[head_index] = surgery_wrapper(self.bbox_head.multi_level_cls_convs[head_index], **kwargs)
            self.bbox_head.multi_level_reg_convs[head_index] = surgery_wrapper(self.bbox_head.multi_level_reg_convs[head_index], **kwargs)
            self.bbox_head.multi_level_conv_cls[head_index] = surgery_wrapper(self.bbox_head.multi_level_conv_cls[head_index], **kwargs)
            self.bbox_head.multi_level_conv_reg[head_index] = surgery_wrapper(self.bbox_head.multi_level_conv_reg[head_index], **kwargs)
            self.bbox_head.multi_level_conv_obj[head_index] = surgery_wrapper(self.bbox_head.multi_level_conv_obj[head_index], **kwargs)
        #
        return self

    def quant_init(self, quant_wrapper, **kwargs):
        self.backbone = quant_wrapper(self.backbone, **kwargs)
        self.neck = quant_wrapper(self.neck, **kwargs)
        # handle self.bbox_head
        num_heads = len(self.bbox_head.multi_level_cls_convs)
        for head_index in range(num_heads):
            self.bbox_head.multi_level_cls_convs[head_index] = quant_wrapper(self.bbox_head.multi_level_cls_convs[head_index], **kwargs)
            self.bbox_head.multi_level_reg_convs[head_index] = quant_wrapper(self.bbox_head.multi_level_reg_convs[head_index], **kwargs)
            self.bbox_head.multi_level_conv_cls[head_index] = quant_wrapper(self.bbox_head.multi_level_conv_cls[head_index], **kwargs)
            self.bbox_head.multi_level_conv_reg[head_index] = quant_wrapper(self.bbox_head.multi_level_conv_reg[head_index], **kwargs)
            self.bbox_head.multi_level_conv_obj[head_index] = quant_wrapper(self.bbox_head.multi_level_conv_obj[head_index], **kwargs)
        #
        return self

    def quant_convert(self, device='cpu', **kwargs):
        self.to(device=device)
        self.backbone = self.backbone.convert()
        self.neck = self.neck.convert()
        # handle self.bbox_head
        num_heads = len(self.bbox_head.multi_level_cls_convs)
        for head_index in range(num_heads):
            self.bbox_head.multi_level_cls_convs[head_index] = self.bbox_head.multi_level_cls_convs[head_index].convert()
            self.bbox_head.multi_level_reg_convs[head_index] = self.bbox_head.multi_level_reg_convs[head_index].convert()
            self.bbox_head.multi_level_conv_cls[head_index] = self.bbox_head.multi_level_conv_cls[head_index].convert()
            self.bbox_head.multi_level_conv_reg[head_index] = self.bbox_head.multi_level_conv_reg[head_index].convert()
            self.bbox_head.multi_level_conv_obj[head_index] = self.bbox_head.multi_level_conv_obj[head_index].convert()
        #
        return self
    
    # def quant_export(self):
    #     return None
