import torch

import torch.distributed
import onnx
from onnxsim import simplify

from  mmengine.dist.utils import  master_only
from .onnx_network import DETR3D_export_model




@master_only
def export_DETR3D(model, inputs=None, data_samples=None, **kwargs):

    onnxModel = DETR3D_export_model(model.img_backbone, 
                                    model.img_neck,
                                    model.pts_bbox_head,
                                    model.add_pred_to_datasample)
    onnxModel.eval()

    # Should clone. Otherwise, when we run both export_model and self.predict,
    # we have error PETRHead forward() - Don't know why
    img = inputs['imgs'].clone()
    batch_img_metas = [ds.metainfo for ds in data_samples]

    batch_img_metas = onnxModel.add_lidar2img(batch_img_metas)
    onnxModel.prepare_data(img, batch_img_metas)

    modelInput = []
    modelInput.append(img)
    model_name = 'detr3d.onnx'

    torch.onnx.export(onnxModel,
                      tuple(modelInput),
                      model_name,
                      opset_version=16,
                      verbose=False)

    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    print("!! ONNX model has been exported for DETR3D!!!\n\n")

