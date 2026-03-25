import torch

import torch.distributed
import onnx
from onnxsim import simplify
from mmengine.dist.utils import  master_only

from .onnx_network import BEVDet_export_model


@master_only
def export_BEVDet(model, inputs=None, data_samples=None, **kwargs):

    onnxModel = BEVDet_export_model(model.img_backbone,
                                    model.img_neck,
                                    model.img_view_transformer,
                                    model.img_bev_encoder_backbone,
                                    model.img_bev_encoder_neck,
                                    model.pts_bbox_head)

    onnxModel.eval()

    img = inputs['imgs'].clone()
    batch_img_metas = [ds.metainfo for ds in data_samples]

    transforms = onnxModel.prepare_data(img, batch_img_metas)
    bev_feat, lidar_coor_1d = onnxModel.precompute_geometry(transforms)

    # save bev_feat and lidar_coor_1d
    bev_feat_np      = bev_feat.to('cpu').numpy()
    lidar_coor_1d_np = lidar_coor_1d.to('cpu').numpy()

    #bev_feat_np.tofile('bevdet_feat.dat')
    #lidar_coor_1d_np.tofile('bevdet_lidar_coor_1d.dat')

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

    modelInput = []
    modelInput.append(img)
    modelInput.append(bev_feat)
    modelInput.append(lidar_coor_1d)

    input_names  = ["inputs", "bev_feat", "lidar_coor_1d"]
    output_names = ["bboxes", "scores", "labels"]

    torch.onnx.export(onnxModel,
                      tuple(modelInput),
                     'bevDet.onnx',
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=16,
                      verbose=False)

    onnx_model, _ = simplify('bevDet.onnx')
    onnx.save(onnx_model, 'bevDet.onnx')

    print("!! ONNX model has been exported for BEVDet!!!\n\n")



