import torch
import copy

from onnxsim import simplify
import onnx
import torch.distributed
import onnx
from onnxsim import simplify

from .onnx_network import PETR_export_model, StreamPETR_export_model, \
                          Far3D_export_model, Far3D_export_img_backbone, Far3D_export_img_roi, Far3D_export_pts_bbox, \
                          BEVFormer_export_model, \
                          BEVDet_export_model, FCOS3D_export_model, \
                          DETR3D_export_model
from  mmengine.dist.utils import is_main_process, is_distributed

def wrap_for_dist(fn):
    def wrapped(*args, **kwargs):
        if is_distributed():
            if is_main_process():
                print('main_process')
                res = fn(*args, **kwargs)
            else:
                print('not main_process')
                res = None
            torch.distributed.barrier()
            return res
        else:
            return fn(*args, **kwargs)
    return wrapped


def export_PETR(model, inputs=None, data_samples=None, 
                quantized_model=False, opset_version=20, **kwargs):

    onnxModel = PETR_export_model(model.img_backbone,
                                  model.img_neck,
                                  model.pts_bbox_head)
    if not quantized_model:
        onnxModel.eval()

    # Should clone. Otherwise, when we run both export_model and self.predict,
    # we have error PETRHead forward() - Don't know why
    img = inputs['imgs'].clone()

    batch_img_metas = [ds.metainfo for ds in data_samples]
    batch_img_metas = onnxModel.add_lidar2img(img, batch_img_metas)
    onnxModel.prepare_data(img, batch_img_metas)
    masks, coords3d = onnxModel.create_coords3d(img)

    # save masks and lidar_coor_1d
    #masks_np    = masks.to('cpu').numpy()
    #coords3d_np = coords3d.to('cpu').numpy()
    #masks_np.tofile('petrv1_masks.dat')
    #coords3d_np.tofile('petrv1_coords3d.dat')

    model_input = []
    model_input.append(img)
    model_input.append(coords3d)

    if quantized_model:
        model_name = 'petrv1_quantized.onnx'
        from edgeai_torchmodelopt import xmodelopt
        xmodelopt.quantization.v3.quant_utils.register_onnx_symbolics(opset_version=opset_version)
    else:
        model_name = 'petrv1.onnx'

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

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

    print("!! ONNX model has been exported for PETR!!!\n\n")



def export_StreamPETR(model, inputs=None, data_samples=None, 
                      opset_version=20, **kwargs):

    onnxModel = StreamPETR_export_model(model.stride,
                                        model.use_grid_mask,
                                        model.grid_mask,
                                        model.img_backbone,
                                        model.img_neck,
                                        model.pts_bbox_head,
                                        model.prepare_location,
                                        model.forward_roi_head)

    onnxModel.eval()

    # Should clone. Otherwise, when we run both export_model and self.predict,
    # we have error PETRHead forward() - Don't know why
    img = inputs['imgs'].clone()

    batch_img_metas = [ds.metainfo for ds in data_samples]
    onnxModel.prepare_data(img, batch_img_metas)
    location = onnxModel.prepare_location(img)

    batch_img_metas[0]['prev_exists'] = img.new_zeros(1)
    x = batch_img_metas[0]['prev_exists'].to(img.device).to(torch.float32)
    memory_embedding, memory_reference_point, memory_timestamp, \
        memory_egopose, memory_velo = onnxModel.pts_bbox_head.init_memory(x)

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

    modelInput = []
    modelInput.append(img)
    modelInput.append(location)
    modelInput.append(memory_embedding)
    modelInput.append(memory_reference_point)
    modelInput.append(memory_timestamp)
    modelInput.append(memory_egopose)
    modelInput.append(memory_velo)

    model_name   = 'streampetr.onnx'
    input_names  = ["imgs", "location", "memory_embed", "memory_ref_point", "memory_ts", "memory_egopose", "memory_velo"]
    output_names = ["bboxes", "scores", "labels"]
    #output_names = ["features"]

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


def export_Far3D_combined(model, inputs=None, data_samples=None,
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
    #locations = onnxModel.prepare_location(img)

    batch_img_metas[0]['prev_exists'] = img.new_zeros(1)
    x = batch_img_metas[0]['prev_exists'].to(img.device).to(torch.float32)
    memory_embedding, memory_reference_point, memory_timestamp, \
        memory_egopose, memory_velo = onnxModel.pts_bbox_head.init_memory(x) #onnxModel.get_memory(x)

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

    model_name   = 'far3d_combined.onnx'
    input_names  = ["imgs", "memory_embed", "memory_ref_point", "memory_ts", "memory_egopose", "memory_velo", "intrinsics", "extrinsics", "lidar2imgs", "img2lidars"]
    output_names = ["bboxes", "scores", "labels"]

    torch.onnx.export(onnxModel,
                      tuple(modelInput),
                      model_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=opset_version,
                      verbose=False)

    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    onnxModel = onnxModel.cuda()

    print("!! ONNX model has been exported for {}!!!\n\n".format(model_name))



def export_Far3D(model, inputs=None, data_samples=None,
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

    intrinsics, extrinsics = onnxModel_img_backbone.prepare_data(img, batch_img_metas)
    batch_img_metas[0]['prev_exists'] = img.new_zeros(1)

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

    modelInput = []
    modelInput.append(img)

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

    intrinsics, extrinsics = onnxModel_img_roi.prepare_data(img_feats[0], batch_img_metas)
    batch_img_metas[0]['prev_exists'] = img.new_zeros(1)

    modelInput = []
    modelInput.append(img_feats)

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
        memory_egopose, memory_velo = onnxModel_pts_bbox.pts_bbox_head.init_memory(x)

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

    model_name   = 'far3d_pts_bbox.onnx'
    input_names  = ["img_feats_0", "img_feats_1", "img_feats_2", "img_feats_3",
                    "pred_depth", "bbox_list_0", "bbox_list_1", "bbox_list_2",
                    "bbox_list_3", "bbox_list_4", "bbox_list_5",
                    "bbox2d_scores", "valid_indices",
                    "memory_embed", "memory_ref_point", "memory_ts", "memory_egopose", "memory_velo", 
                    "intrinsics", "extrinsics", "lidar2imgs", "img2lidars"]
    output_names = ["bboxes", "scores", "labels"]

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

    onnxModel_img_backbone = onnxModel_img_backbone.cuda()
    onnxModel_img_roi = onnxModel_img_roi.cuda()
    onnxModel_pts_bbox = onnxModel_pts_bbox.cuda()



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

    bev_feat_np.tofile('bevdet_feat.dat')
    lidar_coor_1d_np.tofile('bevdet_lidar_coor_1d.dat')

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
    #modelInput.append(data_samples)

    torch.onnx.export(onnxModel,
                      tuple(modelInput),
                     'detr3d.onnx',
                      opset_version=16,
                      verbose=False)

    print("!! ONNX model has been exported for DETR3D!!!\n\n")


def create_onnx_BEVFormer(model):
    onnxModel = BEVFormer_export_model(model.img_backbone,
                                       model.img_neck,
                                       model.pts_bbox_head,
                                       model.add_pred_to_datasample,
                                       model.video_test_mode)
    onnxModel = onnxModel.cpu()
    onnxModel.eval()

    return onnxModel

def export_BEVFormer(onnxModel, inputs=None, data_samples=None, **kwargs):

    # Should clone. Otherwise, when we run both export_model and self.predict,
    img = inputs['imgs'].clone()
    img = img.cpu()

    copy_data_samples = copy.deepcopy(data_samples)
    batch_img_metas = [ds.metainfo for ds in copy_data_samples]

    # For temporal info
    if batch_img_metas[0]['scene_token'] != onnxModel.prev_frame_info['scene_token']:
        onnxModel.prev_frame_info['prev_bev'] = None

    onnxModel.prev_frame_info['scene_token'] = batch_img_metas[0]['scene_token']

    # do not use temporal information
    if not onnxModel.video_test_mode:
        onnxModel.prev_frame_info['prev_bev'] = None

    # Get the delta of ego position and angle between two timestamps.
    tmp_pos = copy.deepcopy(batch_img_metas[0]['can_bus'][:3])
    tmp_angle = copy.deepcopy(batch_img_metas[0]['can_bus'][-1])

    if onnxModel.prev_frame_info['prev_bev'] is not None:
        batch_img_metas[0]['can_bus'][:3] -= onnxModel.prev_frame_info['prev_pos']
        batch_img_metas[0]['can_bus'][-1] -= onnxModel.prev_frame_info['prev_angle']
    else:
        batch_img_metas[0]['can_bus'][-1] = 0
        batch_img_metas[0]['can_bus'][:3] = 0

    if onnxModel.prev_frame_info['prev_bev'] is None:
        onnxModel.prev_frame_info['prev_bev'] = torch.zeros(
            [onnxModel.bev_embedding.weight.size(0), 1, onnxModel.bev_embedding.weight.size(1)])

    # copy batch_img_metas
    onnxModel.prepare_data(batch_img_metas)
    reference_points_cam, bev_mask_count, bev_valid_indices, bev_valid_indices_count, shift_xy, can_bus = \
        onnxModel.precompute_bev_info(batch_img_metas)

    rotation_grid = None
    if onnxModel.prev_frame_info['prev_bev'] is not None:
        rotation_grid = onnxModel.compute_rotation_matrix(
            onnxModel.prev_frame_info['prev_bev'], batch_img_metas)

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

    modelInput = []
    modelInput.append(img)
    modelInput.append(shift_xy)
    modelInput.append(rotation_grid)
    modelInput.append(reference_points_cam)
    modelInput.append(bev_mask_count)
    modelInput.append(bev_valid_indices)
    modelInput.append(bev_valid_indices_count)
    modelInput.append(can_bus)
    modelInput.append(onnxModel.prev_frame_info['prev_bev'])

    input_names  = ["inputs", "shift_xy", "rotation_grid", "reference_points_cam",
                    "bev_mask_count", "bev_valid_indices", "bev_valid_indices_count",
                    "can_bus", "prev_bev"]
    output_names = ["bboxes", "scores", "labels", "bev_feature"]

    torch.onnx.export(onnxModel,
                      tuple(modelInput),
                     'bevFormer.onnx',
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=16,
                      verbose=False)

    onnxModel.prev_frame_info['prev_pos']   = tmp_pos
    onnxModel.prev_frame_info['prev_angle'] = tmp_angle

    onnx_model, _ = simplify('bevFormer.onnx')
    onnx.save(onnx_model, 'bevFormer.onnx')

    # move model back to gpu
    onnxModel = onnxModel.cuda()

    print("!!ONNX model has been exported for BEVFormer!\n\n")

@wrap_for_dist
def     export_FCOS3D(model, inputs=None, data_samples=None, quantized_model=False, opset_version=17):
    onnxModel = FCOS3D_export_model(model.backbone,
                                    model.neck,
                                    model.bbox_head,
                                    model.add_pred_to_datasample)
    
    if not quantized_model:
        onnxModel.eval()

    # Should clone. Otherwise, when we run both export_model and self.predict,
    # we have error PETRHead forward() - Don't know why
    img = inputs['imgs'].clone()
    batch_img_metas = [ds.metainfo for ds in data_samples]

    onnxModel.prepare_data(batch_img_metas)

    cam2img = torch.Tensor(batch_img_metas[0]['cam2img'])
    pad_cam2img = torch.eye(4, dtype=cam2img.dtype).cuda()
    pad_cam2img[:cam2img.shape[0], :cam2img.shape[1]] = cam2img
    inv_pad_cam2img = pad_cam2img.inverse().transpose(0, 1)

    if quantized_model:
        modelInput = []
        modelInput.append(img.cpu())
        modelInput.append(pad_cam2img.cpu())
        modelInput.append(inv_pad_cam2img.cpu())

        from edgeai_torchmodelopt import xmodelopt
        xmodelopt.quantization.v3.quant_utils.register_onnx_symbolics(opset_version=opset_version)

        model_name = 'fcos3d_quantized.onnx'
        
    else:
        modelInput = []
        modelInput.append(img)
        modelInput.append(pad_cam2img)
        modelInput.append(inv_pad_cam2img)

        model_name = 'fcos3d.onnx'

    # Save input & output images
    #fcos3d_img_np  = img.to('cpu').numpy()
    #fcos3d_img_np.tofile('fcos3d_img_np.dat')
    #out = onnxModel(img)
    #for i in range(len(out)):
    #    for j in range(len(out[i])):
    #        out[i][j].to('cpu').numpy().tofile(f"fcos3d_out_{i}_{j}.dat")

    input_names  = ["inputs", "pad_cam2img", "inv_pad_cam2img"]
    output_names = ["mlvl_bboxes", "mlvl_bboxes_for_nms", "mlvl_nms_scores", 
                    "mlvl_dir_scores", "mlvl_attr_scores"]


    torch.onnx.export(onnxModel,
                      tuple(modelInput),
                      model_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=opset_version,
                      training=torch._C._onnx.TrainingMode.PRESERVE,
                      do_constant_folding=False, 
                      verbose=False)

    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    print("!! ONNX model has been exported for FCOS3D! !!\n\n")