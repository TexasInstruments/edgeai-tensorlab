import copy
import onnx
import torch

from onnxsim import simplify
from .onnx_network import VAD_export_model, VAD_export_img_backbone, VAD_export_pts_bbox_head, \
                          precompute_bev_info, compute_rotation_matrix
import numpy as np

def create_onnx_VAD(model):
    # Create a whole model
    onnx_model = VAD_export_model(model.img_backbone,
                                  model.img_neck,
                                  model.pts_bbox_head,
                                  model.video_test_mode,
                                  model.map_pred2result)
    return onnx_model

def create_onnx_VAD_img_backbone(model):
    # Create a img_backbone model
    onnx_model = VAD_export_img_backbone(
                                  model.img_backbone,
                                  model.img_neck)
    return onnx_model

def create_onnx_VAD_pts_bbox_head(model):
    # Create a pts_bbox_head model
    onnx_model = VAD_export_pts_bbox_head(
                                  model.pts_bbox_head,
                                  model.video_test_mode,
                                  model.map_pred2result)
    return onnx_model

def export_VAD_whole(model,
                     inputs,
                     data_samples,
                     opset_version=20,
                     **kwargs):

    if model.onnx_whole is None:
        model.onnx_whole = create_onnx_VAD(model)

    # self.onnx_whole is a whole VAD model
    vad_onnx_whole = model.onnx_whole

    # move model to cpu
    vad_onnx_whole.cpu()
    vad_onnx_whole.eval()

    img = inputs['imgs'].clone().cpu()
    copy_data_samples = copy.deepcopy(data_samples)
    batch_img_metas = [ds.metainfo for ds in copy_data_samples]

    # For temporal info
    if batch_img_metas[0]['scene_token'] != vad_onnx_whole.prev_frame_info['scene_token']:
        vad_onnx_whole.prev_frame_info['prev_bev'] = None
    vad_onnx_whole.prev_frame_info['scene_token'] = batch_img_metas[0]['scene_token']

    # do not use temporal information
    if not vad_onnx_whole.video_test_mode:
        vad_onnx_whole.prev_frame_info['prev_bev'] = None

    # Get the delta of ego position and angle between two timestamps.
    tmp_pos = copy.deepcopy(batch_img_metas[0]['can_bus'][:3])
    tmp_angle = copy.deepcopy(batch_img_metas[0]['can_bus'][-1])

    if vad_onnx_whole.prev_frame_info['prev_bev'] is not None:
        batch_img_metas[0]['can_bus'][:3] -= vad_onnx_whole.prev_frame_info['prev_pos']
        batch_img_metas[0]['can_bus'][-1] -= vad_onnx_whole.prev_frame_info['prev_angle']
    else:
        batch_img_metas[0]['can_bus'][-1] = 0
        batch_img_metas[0]['can_bus'][:3] = 0

    if vad_onnx_whole.prev_frame_info['prev_bev'] is None:
        vad_onnx_whole.prev_frame_info['prev_bev'] = torch.zeros(
            [vad_onnx_whole.bev_embedding.weight.size(0), 1, vad_onnx_whole.bev_embedding.weight.size(1)])

    vad_onnx_whole.set_img_metas(batch_img_metas)
    reference_points_cam, bev_mask_count, bev_valid_indices, bev_valid_indices_count, shift_xy, can_bus = \
        precompute_bev_info(vad_onnx_whole)

    # These tensors are not actuall used in pts_bbox_head forward function for VAD_tiny
    #ego_his_trajs, ego_fut_cmd, ego_lcf_feat = get_ego_features(vad_onnx_whole)

    rotation_grid = None
    if vad_onnx_whole.prev_frame_info['prev_bev'] is not None:
        rotation_grid = compute_rotation_matrix(vad_onnx_whole)

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

    model_input = []
    model_input.append(img)
    model_input.append(shift_xy)
    model_input.append(rotation_grid)
    model_input.append(reference_points_cam)
    model_input.append(bev_mask_count)
    model_input.append(bev_valid_indices)
    model_input.append(bev_valid_indices_count)
    model_input.append(can_bus)
    model_input.append(vad_onnx_whole.prev_frame_info['prev_bev'])

    # Save input tensors
    """
    img_np                     = img.to('cpu').numpy()
    shift_xy_np                = shift_xy.to('cpu').numpy()
    rotation_grid_np           = rotation_grid.to('cpu').numpy()
    reference_points_cam_np    = reference_points_cam.to('cpu').numpy()
    bev_mask_count_np          = bev_mask_count.to('cpu').numpy()
    bev_valid_indices_np       = bev_valid_indices.to('cpu').numpy()
    bev_valid_indices_count_np = bev_valid_indices_count.to('cpu').numpy()
    can_bus_np                 = can_bus.to('cpu').numpy()
    prev_bev_np                = vad_onnx_whole.prev_frame_info['prev_bev'].to('cpu').numpy()

    np.savez('vad_nus.npz', imgs=img_np, shift_xy=shift_xy_np,
             rotation_grid=rotation_grid_np, reference_points_cam=reference_points_cam_np,
             bev_mask_count=bev_mask_count_np, bev_valid_indices=bev_valid_indices_np,
             bev_valid_indices_count=bev_valid_indices_count_np, can_bus=can_bus_np, 
             prev_bev=prev_bev_np)
    """

    model_name   = 'vad_nus.onnx'
    input_names  = ["imgs", "shift_xy", "rotation_grid", "reference_points_cam",
                    "bev_mask_count", "bev_valid_indices", "bev_valid_indices_count",
                    "can_bus", "prev_bev"]
    output_names = ["bboxes", "scores", "labels",
                    "agent_trajs", "map_bboxes", "map_scores", "map_labels",
                    "map_pts", "ego_fut_preds", "bev_feature"]

    torch.onnx.export(vad_onnx_whole,
                      tuple(model_input),
                      model_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=opset_version,
                      verbose=False)

    vad_onnx_whole.prev_frame_info['prev_pos']   = tmp_pos
    vad_onnx_whole.prev_frame_info['prev_angle'] = tmp_angle

    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    # move model back to gpu
    vad_onnx_whole.cuda()

    print("\n!! ONNX model has been exported for VAD!!!\n\n".format(model_name))


def export_VAD_subnets(model,
                       inputs,
                       data_samples,
                       opset_version=20,
                       **kwargs):

    """""""""""""""""""""""""""""""""""
    # VAD_export_img_backbone
    """""""""""""""""""""""""""""""""""
    if model.onnx_img_backbone is None:
        model.onnx_img_backbone = create_onnx_VAD_img_backbone(model)

    # self.onnx_whole is a whole VAD model
    vad_onnx_img_backbone = model.onnx_img_backbone

    vad_onnx_img_backbone = vad_onnx_img_backbone.cpu()
    vad_onnx_img_backbone.eval()

    # Clone img
    img = inputs['imgs'].clone().cpu()
    copy_data_samples = copy.deepcopy(data_samples)
    batch_img_metas = [ds.metainfo for ds in copy_data_samples]

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

    # Export model
    model_input = []
    model_input.append(img)

    # Save input tensors
    """
    img_np                     = img.to('cpu').numpy()
    np.savez('vad_img_backbone_nus.npz', imgs=img_np)
    """

    model_name   = 'vad_img_backbone_nus.onnx'
    input_names  = ["imgs"]
    output_names = ["img_feats"]

    torch.onnx.export(vad_onnx_img_backbone,
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
    # However, in this case, it apperaed somehing messed up and caued error, e.g.,
    # 'Found at least two devices'
    img_feats = vad_onnx_img_backbone.forward(img)
    print("Run image backbone")


    """""""""""""""""""""""""""""""""""
    # VAD_export_pts_bbox_head
    """""""""""""""""""""""""""""""""""
    if model.onnx_pts_bbox_head is None:
        model.onnx_pts_bbox_head = create_onnx_VAD_pts_bbox_head(model)

    # self.onnx_whole is a whole VAD model
    vad_onnx_pts_bbox_head = model.onnx_pts_bbox_head
    vad_onnx_pts_bbox_head = vad_onnx_pts_bbox_head.cpu()
    vad_onnx_pts_bbox_head.eval()

    # For temporal info
    if batch_img_metas[0]['scene_token'] != vad_onnx_pts_bbox_head.prev_frame_info['scene_token']:
        vad_onnx_pts_bbox_head.prev_frame_info['prev_bev'] = None
    vad_onnx_pts_bbox_head.prev_frame_info['scene_token'] = batch_img_metas[0]['scene_token']

    # do not use temporal information
    if not vad_onnx_pts_bbox_head.video_test_mode:
        vad_onnx_pts_bbox_head.prev_frame_info['prev_bev'] = None

    # Get the delta of ego position and angle between two timestamps.
    tmp_pos = copy.deepcopy(batch_img_metas[0]['can_bus'][:3])
    tmp_angle = copy.deepcopy(batch_img_metas[0]['can_bus'][-1])

    if vad_onnx_pts_bbox_head.prev_frame_info['prev_bev'] is not None:
        batch_img_metas[0]['can_bus'][:3] -= vad_onnx_pts_bbox_head.prev_frame_info['prev_pos']
        batch_img_metas[0]['can_bus'][-1] -= vad_onnx_pts_bbox_head.prev_frame_info['prev_angle']
    else:
        batch_img_metas[0]['can_bus'][-1] = 0
        batch_img_metas[0]['can_bus'][:3] = 0

    if vad_onnx_pts_bbox_head.prev_frame_info['prev_bev'] is None:
        vad_onnx_pts_bbox_head.prev_frame_info['prev_bev'] = torch.zeros(
            [vad_onnx_pts_bbox_head.bev_embedding.weight.size(0), 1, 
             vad_onnx_pts_bbox_head.bev_embedding.weight.size(1)])

    vad_onnx_pts_bbox_head.set_img_metas(batch_img_metas)
    reference_points_cam, bev_mask_count, bev_valid_indices, bev_valid_indices_count, shift_xy, can_bus = \
        precompute_bev_info(vad_onnx_pts_bbox_head)

    # These tensors are not actuall used in pts_bbox_head forward function for VAD_tiny
    #ego_his_trajs, ego_fut_cmd, ego_lcf_feat = get_ego_features(vad_onnx_pts_bbox_head)

    rotation_grid = None
    if vad_onnx_pts_bbox_head.prev_frame_info['prev_bev'] is not None:
        rotation_grid = compute_rotation_matrix(vad_onnx_pts_bbox_head)

    # Export model
    model_input = []
    model_input.append(img_feats)
    model_input.append(shift_xy)
    model_input.append(rotation_grid)
    model_input.append(reference_points_cam)
    model_input.append(bev_mask_count)
    model_input.append(bev_valid_indices)
    model_input.append(bev_valid_indices_count)
    model_input.append(can_bus)
    model_input.append(vad_onnx_pts_bbox_head.prev_frame_info['prev_bev'])
    #model_input.append(ego_his_trajs)
    #model_input.append(ego_lcf_feat)

    # Save input tensors
    """
    img_feats_np               = img_feats[0].numpy()
    shift_xy_np                = shift_xy.to('cpu').numpy()
    rotation_grid_np           = rotation_grid.to('cpu').numpy()
    reference_points_cam_np    = reference_points_cam.to('cpu').numpy()
    bev_mask_count_np          = bev_mask_count.to('cpu').numpy()
    bev_valid_indices_np       = bev_valid_indices.to('cpu').numpy()
    bev_valid_indices_count_np = bev_valid_indices_count.to('cpu').numpy()
    can_bus_np                 = can_bus.to('cpu').numpy()
    prev_bev_np                = vad_onnx_pts_bbox_head.prev_frame_info['prev_bev'].to('cpu').numpy()

    np.savez('vad_pts_bbox_head_nus.npz', img_feats=img_feats_np, shift_xy=shift_xy_np,
             rotation_grid=rotation_grid_np, reference_points_cam=reference_points_cam_np,
             bev_mask_count=bev_mask_count_np, bev_valid_indices=bev_valid_indices_np,
             bev_valid_indices_count=bev_valid_indices_count_np, can_bus=can_bus_np, 
             prev_bev=prev_bev_np)
    """

    model_name   = 'vad_pts_bbox_head_nus.onnx'
    input_names  = ["img_feats", "shift_xy", "rotation_grid", "reference_points_cam",
                    "bev_mask_count", "bev_valid_indices", "bev_valid_indices_count",
                    "can_bus", "prev_bev"]
    output_names = ["bboxes", "scores", "labels",
                    "agent_trajs", "map_bboxes", "map_scores", "map_labels",
                    "map_pts", "ego_fut_preds", "bev_feature"]

    torch.onnx.export(vad_onnx_pts_bbox_head,
                      tuple(model_input),
                      model_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=opset_version,
                      verbose=False)

    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)
    print("\n!! ONNX model has been exported for {}!!!\n\n".format(model_name))

    # Run the model
    final_outs = vad_onnx_pts_bbox_head.forward(
                            img_feats, shift_xy, rotation_grid, reference_points_cam,
                            bev_mask_count, bev_valid_indices, bev_valid_indices_count, can_bus,
                            vad_onnx_pts_bbox_head.prev_frame_info['prev_bev'])

    vad_onnx_pts_bbox_head.prev_frame_info['prev_bev']   = final_outs['bev_embed']
    vad_onnx_pts_bbox_head.prev_frame_info['prev_pos']   = tmp_pos
    vad_onnx_pts_bbox_head.prev_frame_info['prev_angle'] = tmp_angle

    print("Run pts_bbox_head")

    # Model back to gpu
    vad_onnx_img_backbone.cuda()
    vad_onnx_pts_bbox_head.cuda()