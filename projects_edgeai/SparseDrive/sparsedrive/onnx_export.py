import copy
import onnx
import torch

from onnxsim import simplify
from .onnx_network import SparseDrive_export_model, \
                          SparseDrive_export_img_backbone, \
                          SparseDrive_export_pts_bbox_head


def create_onnx_SparseDrive(model):
    onnx_model = SparseDrive_export_model(model)

    return onnx_model


def export_SparseDrive(model,
                       inputs=None,
                       data_samples=None,
                       opset_version=16,
                       **kwargs):

    if model.onnx_model is None:
        model.onnx_model = create_onnx_SparseDrive(model)

    # self.onnx_model is a whole SparseDrive model
    sparsedrv_onnx = model.onnx_model

    # move model to cpu
    sparsedrv_onnx.cpu()
    sparsedrv_onnx.eval()

    # Extract img and metadata
    img = inputs['imgs'].clone().cpu()
    batch_img_metas = [ds.metainfo for ds in data_samples]
    sparsedrv_onnx.prepare_data(batch_img_metas)
    time_interval, T_temp2cur = sparsedrv_onnx.get_history(img)

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

    # Extract required tensors from metadata
    projection_mat = []
    gt_ego_fut_cmd = []
    for x in batch_img_metas:
        projection_mat.append(x["projection_mat"])
        gt_ego_fut_cmd.append(x["gt_ego_fut_cmd"])
    projection_mat = torch.stack(projection_mat, dim=0).cpu()
    gt_ego_fut_cmd = torch.cat(gt_ego_fut_cmd, dim=0).cpu()

    model_input = []
    model_input.append(img)
    model_input.append(projection_mat)
    model_input.append(gt_ego_fut_cmd)
    for key, val in sparsedrv_onnx.det_histroy.items():
        model_input.append(val)
    for key, val in sparsedrv_onnx.map_histroy.items():
        model_input.append(val)
    for key, val in sparsedrv_onnx.motion_history.items():
        model_input.append(val)
    model_input.append(time_interval)
    model_input.append(T_temp2cur)

    model_name   = 'sparsedrive.onnx'
    input_names  = ["imgs", "projection_mat", "gt_ego_fut_cmd",
                    "det_cached_feature", "det_cached_anchor", "det_prev_id", "det_instance_id", "det_confidence", "det_temp_confidence",
                    "map_cached_feature", "map_cached_anchor", "map_prev_id", "map_instance_id", "map_confidence", "map_temp_confidence",
                    "motion_prev_instance_id", "motion_prev_confidence", "motion_period", "motion_instance_feature_queue", "motion_anchor_queue",
                    "motion_prev_ego_status", "motion_ego_period", "motion_ego_feature_queue", "motion_ego_anchor_queue",
                    "time_interval", "T_temp2cur"
                    ]
    output_names = ["bboxes_3d", "scores_3d", "labels_3d", "cls_scores", "instance_ids",
                    "map_vectors", "map_scores", "map_labels",
                    "trajs_3d", "trajs_score", "anchor_queue", "period",
                    "planning_score", "planning", "final_planning", "ego_period", "ego_anchor_queue",
                    "det_his_cached_feature", "det_his_cached_anchor", "det_his_prev_id", "det_his_instance_id", "det_his_confidence", "det_his_temp_confidence",
                    "map_his_cached_feature", "map_his_cached_anchor", "map_his_prev_id", "map_his_instance_id", "map_his_confidence", "map_his_temp_confidence",
                    "motion_his_prev_instance_id", "motion_his_prev_confidence", "motion_his_period", "motion_his_instance_feature_queue", "motion_his_anchor_queue",
                    "motion_his_prev_ego_status", "motion_his_ego_period", "motion_his_ego_feature_queue", "motion_his_ego_anchor_queue",
                    ]

    torch.onnx.export(sparsedrv_onnx,
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
    sparsedrv_onnx.cuda()

    # reset the instance bank and queue for inference
    sparsedrv_onnx.pts_bbox_head.det_head.instance_bank.reset()
    sparsedrv_onnx.pts_bbox_head.map_head.instance_bank.reset()
    sparsedrv_onnx.pts_bbox_head.motion_plan_head.instance_queue.reset()

    print("\n!! ONNX model has been exported for SparseDrive!!!\n\n")


def create_onnx_SparseDrive_img_backbone(model):
    onnx_model = SparseDrive_export_img_backbone(model)
    return onnx_model


def create_onnx_SparseDrive_pts_bbox_head(model):
    onnx_model = SparseDrive_export_pts_bbox_head(model)
    return onnx_model


def export_SparseDrive_subnets(model,
                                inputs,
                                data_samples,
                                opset_version=16,
                                **kwargs):
    """""""""""""""""""""""""""
    # Image backbone subnet
    """""""""""""""""""""""""""
    if model.onnx_img_backbone is None:
        model.onnx_img_backbone = create_onnx_SparseDrive_img_backbone(model)

    # self.onnx_model is a whole SparseDrive model
    sparsedrv_onnx_img_backbone = model.onnx_img_backbone

    # move model to cpu
    #sparsedrv_onnx_img_backbone.cpu()
    sparsedrv_onnx_img_backbone.eval()

    # Extract img
    img = inputs['imgs'].clone()
    batch_img_metas = [ds.metainfo for ds in data_samples]
    img_metas = copy.deepcopy(batch_img_metas)

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

    # Export model
    model_input = []
    model_input.append(img)

    model_name   = 'sparsedrive_img_backbone.onnx'
    input_names  = ["imgs"]
    output_names = ["img_feat_0", "img_feat_1", "img_feat_2", "img_feat_3"]

    print("\nExporting {}...".format(model_name))
    torch.onnx.export(sparsedrv_onnx_img_backbone,
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
    # We can run the model after moving to gpu.
    # However, in this case, it appeared something messed up and caused error, e.g.,
    # 'Found at least two devices'
    print("Running image backbone...")
    img_feats = sparsedrv_onnx_img_backbone.forward(img)

    """""""""""""""""""""""""""
    # Head
    """""""""""""""""""""""""""
    if model.onnx_pts_bbox_head is None:
        model.onnx_pts_bbox_head = create_onnx_SparseDrive_pts_bbox_head(model)

    sparsedrv_onnx_head = model.onnx_pts_bbox_head
    sparsedrv_onnx_head = sparsedrv_onnx_head.cpu()
    sparsedrv_onnx_head.eval()

    # move img_feats to cpu
    for i in range(len(img_feats)):
        img_feats[i] = img_feats[i].cpu()

    sparsedrv_onnx_head.prepare_data(batch_img_metas)
    time_interval, T_temp2cur = sparsedrv_onnx_head.get_history(img_feats[0])

    # Extract required tensors from metadata
    projection_mat = []
    #timestamp = []
    gt_ego_fut_cmd = []
    for x in batch_img_metas:
        projection_mat.append(x["projection_mat"])
        #timestamp.append(torch.DoubleTensor([x["timestamp"]]))
        gt_ego_fut_cmd.append(x["gt_ego_fut_cmd"])

    projection_mat = torch.stack(projection_mat, dim=0).cpu()
    #timestamp = torch.cat(timestamp, dim=0).cpu()
    gt_ego_fut_cmd = torch.cat(gt_ego_fut_cmd, dim=0).cpu()

    model_input = []
    model_input.append(img_feats)
    #model_input.append(timestamp.to(dtype=torch.float32))
    model_input.append(projection_mat)
    model_input.append(gt_ego_fut_cmd)
    for key, val in sparsedrv_onnx_head.det_histroy.items():
        model_input.append(val)
    for key, val in sparsedrv_onnx_head.map_histroy.items():
        model_input.append(val)
    for key, val in sparsedrv_onnx_head.motion_history.items():
        model_input.append(val)
    model_input.append(time_interval)
    model_input.append(T_temp2cur)

    model_name   = 'sparsedrive_head.onnx'
    input_names  = ["img_feats_0", "img_feats_1", "img_feats_2", "img_feats_3",
                    "projection_mat", "gt_ego_fut_cmd",
                    "det_cached_feature", "det_cached_anchor", "det_prev_id", "det_instance_id", "det_confidence", "det_temp_confidence",
                    "map_cached_feature", "map_cached_anchor", "map_prev_id", "map_instance_id", "map_confidence", "map_temp_confidence",
                    "motion_prev_instance_id", "motion_prev_confidence", "motion_period", "motion_instance_feature_queue", "motion_anchor_queue",
                    "motion_prev_ego_status", "motion_ego_period", "motion_ego_feature_queue", "motion_ego_anchor_queue",
                    "time_interval", "T_temp2cur"
                    ]
    output_names = ["bboxes_3d", "scores_3d", "labels_3d", "cls_scores", "instance_ids",
                    "map_vectors", "map_scores", "map_labels",
                    "trajs_3d", "trajs_score", "anchor_queue", "period",
                    "planning_score", "planning", "final_planning", "ego_period", "ego_anchor_queue",
                    "det_his_cached_feature", "det_his_cached_anchor", "det_his_prev_id", "det_his_instance_id", "det_his_confidence", "det_his_temp_confidence",
                    "map_his_cached_feature", "map_his_cached_anchor", "map_his_prev_id", "map_his_instance_id", "map_his_confidence", "map_his_temp_confidence",
                    "motion_his_prev_instance_id", "motion_his_prev_confidence", "motion_his_period", "motion_his_instance_feature_queue", "motion_his_anchor_queue",
                    "motion_his_prev_ego_status", "motion_his_ego_period", "motion_his_ego_feature_queue", "motion_his_ego_anchor_queue",
                    ]

    print("\nExporting {}...".format(model_name))
    torch.onnx.export(sparsedrv_onnx_head,
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
    sparsedrv_onnx_head.cuda()

    # reset the instance bank and queue for inference
    sparsedrv_onnx_head.pts_bbox_head.det_head.instance_bank.reset()
    sparsedrv_onnx_head.pts_bbox_head.map_head.instance_bank.reset()
    sparsedrv_onnx_head.pts_bbox_head.motion_plan_head.instance_queue.reset()

    print("\n!! ONNX model has been exported for SparseDrive!!!\n\n")
