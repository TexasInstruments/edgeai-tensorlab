import copy
import onnx
import torch

from onnxsim import simplify
from .onnx_network import SparseDrive_export_model


def create_onnx_SparseDrive(model):
    onnx_model = SparseDrive_export_model(model)

    return onnx_model


def export_SparseDrive(model,
                       opset_version=16,
                       img=None,
                       **kwargs):

    if model.onnx_model is None:
        model.onnx_model = create_onnx_SparseDrive(model)

    # self.onnx_model is a whole MapTR model
    sparsedrv_onnx = model.onnx_model

    # move model to cpu
    sparsedrv_onnx.cpu()
    sparsedrv_onnx.eval()

    nimg = img[0].clone().cpu()
    nimg_metas = copy.deepcopy(kwargs['img_metas'])

    # Passed the squeezed img
    if nimg.dim() == 5 and nimg.size(0) == 1:
        nimg.squeeze_()
    elif nimg.dim() == 5 and nimg.size(0) > 1:
        B, N, C, H, W = img.size()
        nimg = nimg.view(B * N, C, H, W)

    sparsedrv_onnx.prepare_data(nimg_metas)

    model_input = []
    model_input.append(nimg)
    model_input.append(kwargs['timestamp'].to(nimg.device, dtype=torch.float32))
    model_input.append(kwargs['projection_mat'].to(nimg.device))
    model_input.append(kwargs['image_wh'].to(nimg.device))
    model_input.append(kwargs['ego_status'].to(nimg.device))
    model_input.append(kwargs['gt_ego_fut_cmd'].to(nimg.device))

    model_name   = 'sparsedrive.onnx'
    input_names  = ["imgs", "timestamp", "projection_mat", "image_wh",
                    "ego_status", "gt_ego_fut_cmd"]
    #output_names = ["bboxes", "scores", "labels", "map_pts", "bev_feature"]

    torch.onnx.export(sparsedrv_onnx,
                      tuple(model_input),
                      model_name,
                      input_names=input_names,
                      #output_names=output_names,
                      opset_version=opset_version,
                      verbose=False)

    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    # move model back to gpu
    sparsedrv_onnx.cuda()

    print("\n!! ONNX model has been exported for SparseDrive!!!\n\n")

