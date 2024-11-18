
from typing import Dict, List, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmdet.structures import OptSampleList, SampleList
from mmengine.structures import InstanceData
from mmdet.models.utils import (filter_scores_and_topk, select_single_mlvl,
                     unpack_gt_instances)
from torch import Tensor

from mmdet.registry import MODELS
# from mmdet.structures.bbox import bbox2distance
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmdet.models.layers.yolo_layers import Detection 
from mmengine.model import BaseModule
from mmdet.models.utils.yolo_model_utils import LossConfig, MatcherConfig, Vec2Box, NMSConfig, PostProccess

@MODELS.register_module()
class YOLOV9Head(BaseModule):
    def __init__(self, 
                num_classes: int = 80, 
                in_channels: Sequence[int] =[128, 192, 256],
                strides: Sequence[int] = (8, 16, 32),
                #  feat_channels: int = 256,
                reg_max: int = 16,
                use_group: bool = True,
                norm_cfg: OptConfigType = dict(
                    type='GN', num_groups=32, requires_grad=True),
                train_cfg: OptConfigType = None,
                test_cfg: OptConfigType = None,
                loss_yolo: ConfigType = dict(
                    type='YOLOLoss',
                    loss_cfg = LossConfig(
                        objective=dict(
                            BCELoss=0.5,
                            BoxLoss=7.5,
                            DFLoss=1.5
                        ),
                        aux=0.25,
                        matcher=MatcherConfig(
                            iou='CIoU',
                            topk=10,
                            factor=dict(
                                iou=6.0,
                                cls=0.5
                            )
                        )
                    )
                    ),
                    init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu'),
                **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        # super().__init__(
        #     num_classes=num_classes,
        #     in_channels=in_channels,
        #     norm_cfg=norm_cfg,
        #     train_cfg=train_cfg,
        #     test_cfg=test_cfg,
        #     **kwargs)
        self.loss_config = loss_yolo
        self.in_channels = in_channels
        self.strides = strides

        self.heads = nn.ModuleList(
            [Detection((in_channels[0], in_channel), num_classes, reg_max=reg_max) for in_channel in in_channels]
        )
        
    def forward(self, x_lists: List[Tensor]|List[List[Tensor]]) -> List[Tensor]:
        outs = []
        outs.append([head(x) for x, head in zip(x_lists, self.heads)])
        
        return outs

    def loss(self,aux_head, x: Tuple[Tensor], backbone_feat: Tuple[Tensor], batch_data_samples: SampleList, vec2box: Vec2Box) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        predicts = self(x)
        predicts.extend(aux_head(backbone_feat))

        self.loss_config['vec2box'] = vec2box
        self.loss_yolo: nn.Module = MODELS.build(self.loss_config)

        aux_rate = self.loss_config['loss_cfg']['aux']
        iou_rate = self.loss_config['loss_cfg'].objective['BoxLoss']
        dfl_rate = self.loss_config['loss_cfg'].objective['DFLoss']
        cls_rate = self.loss_config['loss_cfg'].objective['BCELoss']

        # TODO:load targets
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs
        
        batch_size = len(batch_gt_instances)
        target_sizes = [item.labels.size(0) for item in batch_gt_instances]
        batch_targets = torch.zeros(batch_size, max(target_sizes), 5).to(batch_gt_instances[0].labels.device)
        batch_targets[:, :, 0] = -1
        for idx, target_size in enumerate(target_sizes):
            label = batch_gt_instances[idx].labels.view(-1,1)
            bbox = batch_gt_instances[idx].bboxes
            target = torch.cat([label,bbox], dim=-1)
            batch_targets[idx][:target_size] = target

        aux_predicts = vec2box(predicts[1])
        main_predicts = vec2box(predicts[0])


        aux_iou, aux_dfl, aux_cls = self.loss_yolo(aux_predicts, batch_targets)
        main_iou, main_dfl, main_cls = self.loss_yolo(main_predicts, batch_targets)

        loss_dict = {
            "loss_box": iou_rate * (aux_iou * aux_rate + main_iou),
            "loss_df": dfl_rate * (aux_dfl * aux_rate + main_dfl),
            "loss_bce": cls_rate * (aux_cls * aux_rate + main_cls),
        }

        return loss_dict


    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                vec2box: Vec2Box,
                nms_cfg: NMSConfig,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(x)
        post_proccess = PostProccess(vec2box, nms_cfg)
        outs = post_proccess(outs)

        result_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            result = InstanceData(
                    bboxes=outs[0][img_id],
                    scores=outs[1][img_id],
                    labels=outs[2][img_id]
                    )
            result_list.append(result)

        return result_list
























##### test ####
def test():
    from mmdet.models.backbones.yolov9_backbone import YOLOV9Backbone
    from mmdet.models.necks.yolov9_neck import YOLOV9Neck
    from torch.onnx import export
    import onnx
    # net = Aconv_RepNCSPELAN(3,64,64,{"repeat_num": 3})
    # net = Yolov9Neck()
    # print(net)
    # x1 = torch.rand(1, 128, 80, 80)
    # x2 = torch.rand(1, 192, 40, 40)
    # x3 = torch.rand(1, 256, 20, 20)
    # x = [x1,x2,x3]
    # y = net(x)
    # print([ys.shape for ys in y])

    net = YOLOV9Backbone()
    net2 = YOLOV9Neck()
    head = YOLOV9Head()

    print(net)
    model = nn.Sequential(net,net2,head)
    x = torch.rand(1,3,640,640)
    # y = net(x)
    # y = net2(y)
    # y = head(y)

    # model = torch.load("work_dirs/onnx_exports/yolov9/checkpoint/yolov9-s.pt")




    y = model(x)
    for yz in y:
        for yx in yz:
            print([ys.shape for ys in yx])
    # onnx_model = torch.onnx.dynamo_export(model,x)
    # onnx_model.save("/data/files/a0508577/work/other/pytorch-tutorial/test_models/yolov9.onnx")
    output_path = "/data/files/a0508577/work/other/pytorch-tutorial/test_models/yolov9.onnx"
    export(
        model,
        x,
        output_path,
        input_names=["input"],
        output_names=["output_names"],
        # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    #simplifier
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    try:
        import onnxsim
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, 'assert check failed'
    except Exception as e:
        print(f'Simplify failure: {e}')
    onnx.save(onnx_model, output_path)

# test()