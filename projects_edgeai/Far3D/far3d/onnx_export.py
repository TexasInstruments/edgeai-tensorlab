import torch

import torch.distributed
import onnx
import numpy as np
from onnxsim import simplify
from mmengine.dist.utils import master_only

from .onnx_network import StreamPETR_export_model, StreamPETR_export_img_backbone, StreamPETR_export_pts_bbox, \
                          Far3D_export_model, Far3D_export_img_backbone, Far3D_export_img_roi, Far3D_export_pts_bbox


@master_only
def export_StreamPETR_whole(model, inputs=None, data_samples=None, 
                      opset_version=20, **kwargs):

    onnxModel = StreamPETR_export_model(model.stride,
                                        model.use_grid_mask,
                                        model.grid_mask,
                                        model.img_backbone,
                                        model.img_neck,
                                        model.pts_bbox_head)
    onnxModel.eval()

    # Should clone. Otherwise, when we run both export_model and self.predict,
    # we have error PETRHead forward() - Don't know why
    img = inputs['imgs'].clone()

    batch_img_metas = [ds.metainfo for ds in data_samples]
    onnxModel.prepare_data(batch_img_metas)
    location = onnxModel.prepare_location(img)
    coords_3d, cone = onnxModel.create_coords3d(location)

    batch_img_metas[0]['prev_exists'] = img.new_zeros(1)
    x = batch_img_metas[0]['prev_exists'].to(img.device).to(torch.float32)

    memory_embedding, memory_reference_point, memory_timestamp, \
        memory_egopose, memory_velo = onnxModel.get_memory(x) #onnxModel.pts_bbox_head.init_memory(x)
    ego_pose, timestamp = onnxModel.get_ego_pose_and_timestamp()
    ego_pose = ego_pose.to(img.device)
    timestamp = timestamp.to(img.device)

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

    modelInput = []
    modelInput.append(img)
    modelInput.append(memory_embedding)
    modelInput.append(memory_reference_point)
    modelInput.append(memory_timestamp)
    modelInput.append(memory_egopose)
    modelInput.append(memory_velo)
    modelInput.append(coords_3d)
    modelInput.append(cone)
    modelInput.append(ego_pose)
    modelInput.append(timestamp)

    # Save input tensors
    """
    img_np                    = img.to('cpu').numpy()
    memory_embedding_np       = memory_embedding.to('cpu').numpy()
    memory_reference_point_np = memory_reference_point.to('cpu').numpy()
    memory_timestamp_np       = memory_timestamp.to('cpu').numpy()
    memory_egopose_np         = memory_egopose.to('cpu').numpy()
    memory_velo_np            = memory_velo.to('cpu').numpy()
    coords_3d_np              = coords_3d.to('cpu').numpy()
    cone_np                   = cone.to('cpu').numpy()
    ego_pose_np               = ego_pose.to('cpu').numpy()
    timestamp_np              = timestamp.to('cpu').numpy()

    np.savez('streampetr_input.npz', imgs=img_np, memory_embed=memory_embedding_np,
             memory_ref_point=memory_reference_point_np, memory_ts=memory_timestamp_np,
             memory_egopose=memory_egopose_np, memory_velo=memory_velo_np,
             coords_3d=coords_3d_np, cone=cone_np, ego_pose=ego_pose_np, timestamp=timestamp_np)
    """

    model_name   = 'streampetr.onnx'
    input_names  = ["imgs", "memory_embed", "memory_ref_point", "memory_ts",
                    "memory_egopose", "memory_velo", 
                    "coords_3d", "cone", "ego_pose", "timestamp"]
    output_names = ["bboxes", "scores", "labels",
                    "out_memory_embed", "out_memory_ref_point", "out_memory_ts",
                    "out_memory_egopose", "out_memory_velo"]

    torch.onnx.export(onnxModel,
                      tuple(modelInput),
                      model_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=opset_version,
                      verbose=False)

    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    print("!! ONNX model has been exported for {}!!!\n\n".format(model_name))


@master_only
def export_StreamPETR_subnets(model, inputs=None, data_samples=None, 
                      opset_version=20, **kwargs):

    """""""""""""""""""""""""""""""""""
    # StreamPETR_img_backbone
    """""""""""""""""""""""""""""""""""
    onnxModel_img_backbone = StreamPETR_export_img_backbone(
                                   model.use_grid_mask,
                                   model.grid_mask,
                                   model.img_backbone,
                                   model.img_neck)
    onnxModel_img_backbone = onnxModel_img_backbone.cpu()
    onnxModel_img_backbone.eval()

    # Should clone. Otherwise, when we run both export_model and self.predict,
    # we have error PETRHead forward() - Don't know why
    img = inputs['imgs'].clone()
    img = img.cpu()
    batch_img_metas = [ds.metainfo for ds in data_samples]

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

    modelInput = []
    modelInput.append(img)
    
    model_name   = 'streampetr_img_backbone.onnx'
    input_names  = ["imgs"]
    output_names = ["img_faats"]

    torch.onnx.export(onnxModel_img_backbone,
                      tuple(modelInput),
                      model_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=opset_version,
                      verbose=False)

    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    print("!! ONNX model has been exported for {}!!!\n\n".format(model_name))

    img_feats = onnxModel_img_backbone.forward(img)
    print("Run image backbone")

    """""""""""""""""""""""""""""""""""
    # StreamPETR_pts_bbox
    """""""""""""""""""""""""""""""""""
    onnxModel_pts_bbox = StreamPETR_export_pts_bbox(
                                   model.stride,
                                   model.pts_bbox_head)
    onnxModel_pts_bbox = onnxModel_pts_bbox.cpu()
    onnxModel_pts_bbox.eval()

    onnxModel_pts_bbox.prepare_data(batch_img_metas)
    location = onnxModel_pts_bbox.prepare_location(img)
    coords_3d, cone = onnxModel_pts_bbox.create_coords3d(location)

    batch_img_metas[0]['prev_exists'] = img.new_zeros(1)
    x = batch_img_metas[0]['prev_exists'].to(img.device).to(torch.float32)

    memory_embedding, memory_reference_point, memory_timestamp, \
        memory_egopose, memory_velo = onnxModel_pts_bbox.get_memory(x) #onnxModel.pts_bbox_head.init_memory(x)
    ego_pose, timestamp = onnxModel_pts_bbox.get_ego_pose_and_timestamp()
    ego_pose = ego_pose.to(img.device)
    timestamp = timestamp.to(img.device)

    modelInput = []
    modelInput.append(img_feats)
    modelInput.append(memory_embedding)
    modelInput.append(memory_reference_point)
    modelInput.append(memory_timestamp)
    modelInput.append(memory_egopose)
    modelInput.append(memory_velo)
    modelInput.append(coords_3d)
    modelInput.append(cone)
    modelInput.append(ego_pose)
    modelInput.append(timestamp)

    # Save input tensors
    """
    img_feats_np              = img_feats.to('cpu').numpy()
    memory_embedding_np       = memory_embedding.to('cpu').numpy()
    memory_reference_point_np = memory_reference_point.to('cpu').numpy()
    memory_timestamp_np       = memory_timestamp.to('cpu').numpy()
    memory_egopose_np         = memory_egopose.to('cpu').numpy()
    memory_velo_np            = memory_velo.to('cpu').numpy()
    coords_3d_np              = coords_3d.to('cpu').numpy()
    cone_np                   = cone.to('cpu').numpy()
    ego_pose_np               = ego_pose.to('cpu').numpy()
    timestamp_np              = timestamp.to('cpu').numpy()

    np.savez('streampetr_pts_bbox_input.npz', img_feats=img_feats_np, memory_embed=memory_embedding_np,
             memory_ref_point=memory_reference_point_np, memory_ts=memory_timestamp_np,
             memory_egopose=memory_egopose_np, memory_velo=memory_velo_np,
             coords_3d=coords_3d_np, cone=cone_np, ego_pose=ego_pose_np, timestamp=timestamp_np)
    """

    model_name   = 'streampetr_pts_bbox.onnx'
    input_names  = ["img_feats", "memory_embed", "memory_ref_point", "memory_ts",
                    "memory_egopose", "memory_velo", 
                    "coords_3d", "cone", "ego_pose", "timestamp"]
    output_names = ["bboxes", "scores", "labels",
                    "out_memory_embed", "out_memory_ref_point", "out_memory_ts",
                    "out_memory_egopose", "out_memory_velo"]

    torch.onnx.export(onnxModel_pts_bbox,
                      tuple(modelInput),
                      model_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=opset_version,
                      verbose=False)

    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    print("!! ONNX model has been exported for {}!!!\n\n".format(model_name))

    final_outs = onnxModel_pts_bbox.forward(img_feats, memory_embedding, memory_reference_point, memory_timestamp,
                                            memory_egopose, memory_velo, coords_3d, cone, ego_pose, timestamp)
    print("Run pts_bbox_head")

    # Move the models to GPU
    onnxModel_img_backbone.cuda()
    onnxModel_pts_bbox.cuda()


@master_only
def export_Far3D_whole(model, inputs=None, data_samples=None,
                       opset_version=20, **kwargs):

    onnxModel = Far3D_export_model(model.stride,
                                   model.use_grid_mask,
                                   model.with_img_neck,
                                   model.single_test,
                                   model.position_level,
                                   model.aux_2d_only,
                                   model.grid_mask,
                                   model.img_backbone,
                                   model.img_neck,
                                   model.pts_bbox_head,
                                   model.img_roi_head)
    onnxModel = onnxModel.cpu()
    onnxModel.eval()

    # Should clone. Otherwise, when we run both export_model and self.predict,
    # we have error PETRHead forward() - Don't know why
    img = inputs['imgs'].clone()
    img = img.cpu()

    batch_img_metas = [ds.metainfo for ds in data_samples]
    intrinsics, extrinsics, lidar2imgs, img2lidars = onnxModel.prepare_data(img, batch_img_metas)

    batch_img_metas[0]['prev_exists'] = img.new_zeros(1)
    x = batch_img_metas[0]['prev_exists'].to(img.device).to(torch.float32)

    memory_embedding, memory_reference_point, memory_timestamp, \
        memory_egopose, memory_velo = onnxModel.get_memory(x) # onnxModel.pts_bbox_head.init_memory(x)
    ego_pose, timestamp = onnxModel.get_ego_pose_and_timestamp()
    ego_pose = ego_pose.to(img.device)
    timestamp = timestamp.to(img.device)

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

    modelInput = []
    modelInput.append(img)
    modelInput.append(memory_embedding)
    modelInput.append(memory_reference_point)
    modelInput.append(memory_timestamp)
    modelInput.append(memory_egopose)
    modelInput.append(memory_velo)
    modelInput.append(intrinsics)
    modelInput.append(extrinsics)
    modelInput.append(lidar2imgs)
    modelInput.append(img2lidars)
    modelInput.append(ego_pose)
    modelInput.append(timestamp)

    # Save input tensors
    """
    img_np                    = img.to('cpu').numpy()
    memory_embedding_np       = memory_embedding.to('cpu').numpy()
    memory_reference_point_np = memory_reference_point.to('cpu').numpy()
    memory_timestamp_np       = memory_timestamp.to('cpu').numpy()
    memory_egopose_np         = memory_egopose.to('cpu').numpy()
    memory_velo_np            = memory_velo.to('cpu').numpy()
    intrinsics_np             = intrinsics.to('cpu').numpy()
    extrinsics_np             = extrinsics.to('cpu').numpy()
    lidar2imgs_np             = lidar2imgs.to('cpu').numpy()
    img2lidars_np             = img2lidars.to('cpu').numpy()
    ego_pose_np               = ego_pose.to('cpu').numpy()
    timestamp_np              = timestamp.to('cpu').numpy()

    np.savez('far3d_input.npz', imgs=img_np, memory_embed=memory_embedding_np,
             memory_ref_point=memory_reference_point_np, memory_ts=memory_timestamp_np,
             memory_egopose=memory_egopose_np, memory_velo=memory_velo_np,
             intrinsics=intrinsics_np, extrinsics=extrinsics_np, lidar2imgs=lidar2imgs_np, img2lidars=img2lidars_np,
             ego_pose=ego_pose_np, timestamp=timestamp_np
             )
    """

    model_name   = 'far3d.onnx'
    input_names  = ["imgs", "memory_embed", "memory_ref_point", "memory_ts", "memory_egopose", "memory_velo",
                    "intrinsics", "extrinsics", "lidar2imgs", "img2lidars",
                    "ego_pose", "timestamp"]
    output_names = ["bboxes", "scores", "labels",
                    "out_memory_embed", "out_memory_ref_point", "out_memory_ts",
                    "out_memory_egopose", "out_memory_velo"]

    torch.onnx.export(onnxModel,
                      tuple(modelInput),
                      model_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=opset_version,
                      verbose=False)

    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    onnxModel.cuda()
    print("!! ONNX model has been exported for {}!!!\n\n".format(model_name))


@master_only
def export_Far3D_subnets(model, inputs=None, data_samples=None,
                         opset_version=20, **kwargs):

    """""""""""""""""""""""""""""""""""
    # Far3D_export_model_img_backbone
    """""""""""""""""""""""""""""""""""
    onnxModel_img_backbone = Far3D_export_img_backbone(
                                   model.use_grid_mask,
                                   model.with_img_neck,
                                   model.position_level,
                                   model.grid_mask,
                                   model.img_backbone,
                                   model.img_neck)
    onnxModel_img_backbone = onnxModel_img_backbone.cpu()
    onnxModel_img_backbone.eval()

    # Should clone. Otherwise, when we run both export_model and self.predict,
    # we have error PETRHead forward() - Don't know why
    img = inputs['imgs'].clone()
    img = img.cpu()
    batch_img_metas = [ds.metainfo for ds in data_samples]

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

    modelInput = []
    modelInput.append(img)

    # Save input tensors
    """
    img_np                    = img.to('cpu').numpy()
    np.savez('far3d_img_backbone_input.npz', imgs=img_np)
    """

    model_name   = 'far3d_img_backbone.onnx'
    input_names  = ["imgs"]
    output_names = ["img_feats_0", "img_feats_1", "img_feats_2", "img_feats_3"]

    torch.onnx.export(onnxModel_img_backbone,
                      tuple(modelInput),
                      model_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=opset_version,
                      verbose=False)

    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)
    print("!! ONNX model has been exported for {}!!!\n\n".format(model_name))

    img_feats = onnxModel_img_backbone.forward(img)
    print("Run image backbone")


    """""""""""""""""""""""""""""""""""
    # Far3D_export_img_roi
    """""""""""""""""""""""""""""""""""
    onnxModel_img_roi = Far3D_export_img_roi(model.img_roi_head)
    onnxModel_img_roi = onnxModel_img_roi.cpu()
    onnxModel_img_roi.eval()

    modelInput = []
    modelInput.append(img_feats)

    # Save input tensors
    """
    img_feats_np = []
    for idx, img_feat in enumerate(img_feats):
        img_feats_np.append(img_feat.to('cpu').numpy())
    np.savez('far3d_img_roi_input.npz', img_feats_0=img_feats_np[0], img_feats_1=img_feats_np[1],
             img_feats_2=img_feats_np[2], img_feats_3=img_feats_np[3])
    """

    model_name   = 'far3d_img_roi.onnx'
    input_names  = ["img_feats_0", "img_feats_1", "img_feats_2", "img_feats_3"]
    output_names = ["pred_depth", "bbox_list_0", "bbox_list_1", "bbox_list_2",
                    "bbox_list_3", "bbox_list_4", "bbox_list_5",
                    "bbox2d_scores", "valid_indices"]

    torch.onnx.export(onnxModel_img_roi,
                      tuple(modelInput),
                      model_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=opset_version,
                      verbose=False)

    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)
    print("!! ONNX model has been exported for {}!!!\n\n".format(model_name))

    pred_depth, bbox_list, bbox2d_scores, valid_indices = onnxModel_img_roi.forward(img_feats)
    print("Run image ROI")

    """""""""""""""""""""""""""""""""""
    # Far3D_export_pts_bbox
    """""""""""""""""""""""""""""""""""
    onnxModel_pts_bbox = Far3D_export_pts_bbox(model.pts_bbox_head)
    onnxModel_pts_bbox = onnxModel_pts_bbox.cpu()
    onnxModel_pts_bbox.eval()

    intrinsics, extrinsics, lidar2imgs, img2lidars = onnxModel_pts_bbox.prepare_data(img_feats[0], batch_img_metas)
    batch_img_metas[0]['prev_exists'] = img_feats[0].new_zeros(1)
    x = batch_img_metas[0]['prev_exists'].to(img_feats[0].device).to(torch.float32)

    memory_embedding, memory_reference_point, memory_timestamp, \
        memory_egopose, memory_velo = onnxModel_pts_bbox.get_memory(x)  # onnxModel_pts_bbox.pts_bbox_head.init_memory(x)
    ego_pose, timestamp = onnxModel_pts_bbox.get_ego_pose_and_timestamp()
    ego_pose = ego_pose.to(img.device)
    timestamp = timestamp.to(img.device)

    outs_roi = {
        'pred_depth': pred_depth, 
        'bbox_list': bbox_list,
        'bbox2d_scores' : bbox2d_scores,
        'valid_indices' : valid_indices
    }

    modelInput = []
    modelInput.append(img_feats)
    modelInput.append(outs_roi)
    modelInput.append(memory_embedding)
    modelInput.append(memory_reference_point)
    modelInput.append(memory_timestamp)
    modelInput.append(memory_egopose)
    modelInput.append(memory_velo)
    modelInput.append(intrinsics)
    modelInput.append(extrinsics)
    modelInput.append(lidar2imgs)
    modelInput.append(img2lidars)
    modelInput.append(ego_pose)
    modelInput.append(timestamp)

    # Save input tensors
    img_feats_np = []
    for idx, img_feat in enumerate(img_feats):
        img_feats_np.append(img_feat.to('cpu').numpy())
    outs_roi_np = {}
    bbox_list_np = []
    for key, val in outs_roi.items():
        if key == 'bbox_list':
            for idx, bbox_list in enumerate(outs_roi[key]):
                bbox_list_np.append(bbox_list.to('cpu').numpy())
        else:
            outs_roi_np[key] = val.to('cpu').numpy()

    """
    memory_embedding_np       = memory_embedding.to('cpu').numpy()
    memory_reference_point_np = memory_reference_point.to('cpu').numpy()
    memory_timestamp_np       = memory_timestamp.to('cpu').numpy()
    memory_egopose_np         = memory_egopose.to('cpu').numpy()
    memory_velo_np            = memory_velo.to('cpu').numpy()
    intrinsics_np             = intrinsics.to('cpu').numpy()
    extrinsics_np             = extrinsics.to('cpu').numpy()
    lidar2imgs_np             = lidar2imgs.to('cpu').numpy()
    img2lidars_np             = img2lidars.to('cpu').numpy()
    ego_pose_np               = ego_pose.to('cpu').numpy()
    timestamp_np              = timestamp.to('cpu').numpy()

    np.savez('far3d_pts_bbox_input.npz', img_feats_0=img_feats_np[0], img_feats_1=img_feats_np[1],
             img_feats_2=img_feats_np[2], img_feats_3=img_feats_np[3],
             pred_depth=outs_roi_np['pred_depth'], bbox_list_0=bbox_list_np[0],
             bbox_list_1=bbox_list_np[1], bbox_list_2=bbox_list_np[2],
             bbox_list_3=bbox_list_np[3], bbox_list_4=bbox_list_np[4],
             bbox_list_5=bbox_list_np[5], bbox2d_scores=outs_roi_np['bbox2d_scores'],
             valid_indices=outs_roi_np['valid_indices'],
             memory_embed=memory_embedding_np,
             memory_ref_point=memory_reference_point_np, memory_ts=memory_timestamp_np,
             memory_egopose=memory_egopose_np, memory_velo=memory_velo_np,
             intrinsics=intrinsics_np, extrinsics=extrinsics_np, lidar2imgs=lidar2imgs_np, img2lidars=img2lidars_np,
             ego_pose=ego_pose_np, timestamp=timestamp_np)
    """

    model_name   = 'far3d_pts_bbox.onnx'
    input_names  = ["img_feats_0", "img_feats_1", "img_feats_2", "img_feats_3",
                    "pred_depth", "bbox_list_0", "bbox_list_1", "bbox_list_2",
                    "bbox_list_3", "bbox_list_4", "bbox_list_5",
                    "bbox2d_scores", "valid_indices",
                    "memory_embed", "memory_ref_point", "memory_ts", "memory_egopose", "memory_velo", 
                    "intrinsics", "extrinsics", "lidar2imgs", "img2lidars",
                    "ego_pose", "timestamp"]
    output_names = ["bboxes", "scores", "labels",
                    "out_memory_embed", "out_memory_ref_point", "out_memory_ts",
                    "out_memory_egopose", "out_memory_velo"]

    if model.img_roi_head.sample_with_score is True:
        dynamic_axes = {"bbox_list_0": {0: "bbox_num_0"},
                        "bbox_list_1": {0: "bbox_num_1"},
                        "bbox_list_2": {0: "bbox_num_2"},
                        "bbox_list_3": {0: "bbox_num_3"},
                        "bbox_list_4": {0: "bbox_num_4"},
                        "bbox_list_5": {0: "bbox_num_5"},
                        "bbox2d_scores": {0: "bbox_num_total"}
                       }
    else:
        dynamic_axes = None
    torch.onnx.export(onnxModel_pts_bbox,
                      tuple(modelInput),
                      model_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=opset_version,
                      dynamic_axes=dynamic_axes,
                      verbose=False)

    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)
    print("!! ONNX model has been exported for {}!!!\n\n".format(model_name))

    final_outs = onnxModel_pts_bbox.forward(img_feats, outs_roi, memory_embedding, memory_reference_point, memory_timestamp,
                                            memory_egopose, memory_velo, intrinsics, extrinsics, lidar2imgs, img2lidars,
                                            ego_pose, timestamp)
    print("Run pts_bbox_head")

    onnxModel_img_backbone.cuda()
    onnxModel_img_roi.cuda()
    onnxModel_pts_bbox.cuda()



