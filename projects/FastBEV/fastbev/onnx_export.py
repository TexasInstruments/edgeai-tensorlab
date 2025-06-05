import torch
import copy

from onnxsim import simplify
import onnx
from .onnx_network import FastBEV_export_model

from  mmengine.dist.utils import  master_only

@master_only
def create_onnx_FastBEV(model, quantized_model=False):
    onnxModel = FastBEV_export_model(model)
    onnxModel = onnxModel.cpu()
    if not quantized_model:
        onnxModel.eval()

    return onnxModel

@master_only
def export_FastBEV(onnxModel, inputs=None, data_samples=None,
                   quantized_model=False, opset_version=20,  **kwargs):

    img = inputs['imgs'].clone()
    img = img.cpu()
    batch_img_metas = [ds.metainfo for ds in data_samples]

    onnxModel.prepare_data(batch_img_metas)

    prev_feats_map = None
    prev_input_metas = None

    if onnxModel.num_temporal_feats > 0:
        prev_feats_map, prev_input_metas = onnxModel.get_temporal_feats(
            onnxModel.queue, onnxModel.memory, img, batch_img_metas,
            onnxModel.feats_size, onnxModel.num_temporal_feats)

    xy_coors = onnxModel.precompute_proj_info_for_inference(img, batch_img_metas, prev_img_metas=prev_input_metas)

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

    model_input = []
    model_input.append(img)
    model_input.append(xy_coors)
    model_input.append(prev_feats_map)

    if quantized_model:
        from edgeai_torchmodelopt import xmodelopt
        xmodelopt.quantization.v3.quant_utils.register_onnx_symbolics(opset_version=opset_version)
        model_name = f"fastbev_f{onnxModel.num_temporal_feats+1}_quantized.onnx"
    else:
        model_name = f"fastbev_f{onnxModel.num_temporal_feats+1}.onnx"

    input_names  = ["imgs", "xy_coor", "prev_img_feats"]
    if prev_feats_map is None:
        output_names = ["bboxes", "scores", "labels"]
        #output_names = ["bboxes", "bboxes_for_nms", "scores", "dir_scores"]
    else:
        output_names = ["bboxes", "scores", "labels", "img_feats"]
        #output_names = ["bboxes", "bboxes_for_nms", "scores",  "dir_scores", "img_feats"]

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

    # move model back to gpu
    onnxModel = onnxModel.cuda()

    print("\n!! ONNX model has been exported for FastBEV!!!\n\n")

