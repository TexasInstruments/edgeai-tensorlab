import copy
import onnx
import torch

from onnxsim import simplify
from .onnx_network import Sparse4D_export_model, \
                          Sparse4D_export_img_backbone, \
                          Sparse4D_export_head


def create_onnx_Sparse4D(model):
    onnx_model = Sparse4D_export_model(model)
    return onnx_model

def export_Sparse4D(model,
                    inputs=None,
                    data_samples=None,
                    opset_version=18,
                    **kwargs):

    if model.onnx_model is None:
        model.onnx_model = create_onnx_Sparse4D(model)

    # self.onnx_model is a whole Sparse4D model
    sparse4d_onnx = model.onnx_model
    sparse4d_onnx.eval()

    img = inputs['imgs'].clone()
    batch_img_metas = [ds.metainfo for ds in data_samples]
    sparse4d_onnx.prepare_data(batch_img_metas)
    time_interval, T_temp2cur = sparse4d_onnx.get_history(img)
    time_interval = torch.Tensor([time_interval]).to(img.device)

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        b, n, c, h, w = img.size()
        img = img.view(b * n, c, h, w)

    model_input = []
    model_input.append(img)
    model_input.append(batch_img_metas[0]['projection_mat'].to(img.device))
    for key, val in sparse4d_onnx.det_histroy.items():
        model_input.append(val)
    model_input.append(time_interval)
    model_input.append(T_temp2cur)

    model_name   = 'sparse4d.onnx'
    input_names  = ["imgs", "projection_mat",
                    "det_cached_feature", "det_cached_anchor", "det_prev_id", "det_instance_id", "det_confidence", "det_temp_confidence",
                    "time_interval", "T_temp2cur"]
    output_names = ["bboxes_3d", "scores_3d", "labels_3d", "cls_scores", "instance_ids",
                    "det_his_cached_feature", "det_his_cached_anchor", "det_his_prev_id", "det_his_instance_id", "det_his_confidence", "det_his_temp_confidence"]

    torch.onnx.export(sparse4d_onnx,
                      tuple(model_input),
                      model_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=opset_version,
                      verbose=False)

    print("Model exported. Being simplified ...")
    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    # move model back to gpu
    sparse4d_onnx.cuda()

    # reset the instance bank for inference
    sparse4d_onnx.pts_bbox_head.instance_bank.reset()

    print("\n!! ONNX model has been exported for Sparse4D!!!\n\n")


def create_onnx_Sparse4D_img_backbone(model):
    onnx_model = Sparse4D_export_img_backbone(model)
    return onnx_model


def create_onnx_Sparse4D_head(model):
    onnx_model = Sparse4D_export_head(model)
    return onnx_model


def export_Sparse4D_subnets(model,
                            inputs=None,
                            data_samples=None,
                            opset_version=18,
                            **kwargs):
    """""""""""""""""""""""""""
    # Image backbone subnet
    """""""""""""""""""""""""""
    if model.onnx_img_backbone is None:
        model.onnx_img_backbone = create_onnx_Sparse4D_img_backbone(model)

    sparse4d_onnx_img_backbone = model.onnx_img_backbone
    sparse4d_onnx_img_backbone.eval()

    img = inputs['imgs'].clone()
    batch_img_metas = [ds.metainfo for ds in data_samples]

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        b, n, c, h, w = img.size()
        img = img.view(b * n, c, h, w)

    model_input = []
    model_input.append(img)
    model_name   = 'sparse4d_img_backbone.onnx'
    input_names  = ["imgs"]
    output_names = ["img_feat_0", "img_feat_1", "img_feat_2", "img_feat_3"]

    print("\nExporting {}...".format(model_name))
    torch.onnx.export(sparse4d_onnx_img_backbone,
                      tuple(model_input),
                      model_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=opset_version,
                      verbose=False)

    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)
    print("!! ONNX model has been exported for {}!!!\n\n".format(model_name))

    # Run the model
    print("Running image backbone...")
    img_feats = sparse4d_onnx_img_backbone.forward(img)

    """""""""""""""""""""""""""
    # pts_bbox_head
    """""""""""""""""""""""""""
    if model.onnx_pts_bbox_head is None:
        model.onnx_pts_bbox_head = create_onnx_Sparse4D_head(model)

    sparse4d_onnx_head = model.onnx_pts_bbox_head
    sparse4d_onnx_head.eval()

    sparse4d_onnx_head.prepare_data(batch_img_metas)
    time_interval, T_temp2cur = sparse4d_onnx_head.get_history(img_feats[0])
    time_interval = torch.Tensor([time_interval]).to(img_feats[0].device)

    model_input = []
    model_input.append(img_feats)
    model_input.append(batch_img_metas[0]['projection_mat'].to(img.device))
    for key, val in sparse4d_onnx_head.det_histroy.items():
        model_input.append(val)
    model_input.append(time_interval)
    model_input.append(T_temp2cur)

    model_name   = 'sparse4d_pts_bbox_head.onnx'
    input_names  = ["img_feats_0", "img_feats_1", "img_feats_2", "img_feats_3",
                    "projection_mat",
                    "det_cached_feature", "det_cached_anchor", "det_prev_id", "det_instance_id", "det_confidence", "det_temp_confidence",
                    "time_interval", "T_temp2cur"]
    output_names = ["bboxes_3d", "scores_3d", "labels_3d", "cls_scores", "instance_ids",
                    "det_his_cached_feature", "det_his_cached_anchor", "det_his_prev_id", "det_his_instance_id", "det_his_confidence", "det_his_temp_confidence"]

    print("\nExporting {}...".format(model_name))
    torch.onnx.export(sparse4d_onnx_head,
                      tuple(model_input),
                      model_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=opset_version,
                      verbose=False)

    print("Model exported. Being simplified ...")
    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    # reset the instance bank and queue for inference
    sparse4d_onnx_head.pts_bbox_head.instance_bank.reset()
    print("\n!! ONNX model has been exported for SparseDrive!!!\n\n")
