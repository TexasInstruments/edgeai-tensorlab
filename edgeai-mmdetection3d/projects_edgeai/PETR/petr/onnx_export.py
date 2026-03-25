import torch
import copy

import torch.distributed
import onnx
from onnxsim import simplify

from .onnx_network import PETR_export_model
from  mmengine.dist.utils import  master_only

@master_only
def export_PETR(model, inputs=None, data_samples=None,
                quantized_model=False, opset_version=20, **kwargs):

    onnxModel = PETR_export_model(model,
                                  model.img_backbone,
                                  model.img_neck,
                                  model.pts_bbox_head)
    if not quantized_model:
        onnxModel.eval()

    # Should clone. Otherwise, when we run both export_model and self.predict,
    # we have error PETRHead forward() - Don't know why
    img = inputs['imgs'].clone()

    # PETRv2 update batch_img_metas,
    # so use the copied batch_img_metas
    batch_img_metas_org = [ds.metainfo for ds in data_samples]
    batch_img_metas = copy.deepcopy(batch_img_metas_org)

    prev_feats_map = 0
    if model.version == 'v2':
        prev_feats_map, batch_img_metas = onnxModel.get_temporal_feats(
            onnxModel.queue, onnxModel.memory, img, batch_img_metas, onnxModel.img_feat_size)
        if torch.is_tensor(prev_feats_map) is False:
            prev_feats_map = torch.zeros([1]+model.img_feat_size[0],dtype=img.dtype,device=img.device)
            is_prev_feat = 0
        else:
            is_prev_feat = 1
        is_prev_feat = torch.tensor(is_prev_feat, dtype=torch.int32, device=img.device)

    batch_img_metas = onnxModel.add_lidar2img(img, batch_img_metas)
    onnxModel.prepare_data(img, batch_img_metas)
    masks, coords3d = onnxModel.create_coords3d(img)

    # save masks and lidar_coor_1d
    #masks_np    = masks.to('cpu').numpy()
    #coords3d_np = coords3d.to('cpu').numpy()
    #masks_np.tofile('petrv1_masks.dat')
    #coords3d_np.tofile('petrv1_coords3d.dat')

    # Set inputs and outputs
    model_input = []
    model_input.append(img)
    model_input.append(coords3d)
    if model.version == 'v2':
        model_input.append(is_prev_feat)
        model_input.append(prev_feats_map)

    # Set the model name
    model_name = 'petrv1.onnx'
    if model.version == 'v2':
        model_name = 'petrv2.onnx'

    if quantized_model:
        model_name = 'petrv1_quantized.onnx'
        if model.version == 'v2':
            model_name = 'petrv2_quantized.onnx'
        from edgeai_torchmodelopt import xmodelopt
        xmodelopt.quantization.v3.quant_utils.register_onnx_symbolics(opset_version=opset_version)

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

    if model.version == 'v2':
        input_names  = ["imgs", "coords3d", "valid_prev_feats", "prev_feats_map"]
        output_names = ["bboxes", "scores", "labels", 'feats_map']
    else:
        input_names  = ["imgs", "coords3d"]
        output_names = ["bboxes", "scores", "labels"]

    torch.onnx.export(onnxModel,
                      tuple(model_input),
                      model_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=opset_version,
                      training=torch._C._onnx.TrainingMode.PRESERVE,
                      verbose=False)

    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    print("!! ONNX model has been exported for PETR{}!!!\n\n".format(model.version))


