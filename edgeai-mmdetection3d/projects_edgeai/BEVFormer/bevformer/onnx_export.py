import copy
import torch

import torch.distributed
import onnx
from onnxsim import simplify
from  mmengine.dist.utils import  master_only

from .onnx_network import BEVFormer_export_model


@master_only
def create_onnx_BEVFormer(model):
    onnxModel = BEVFormer_export_model(model.img_backbone,
                                       model.img_neck,
                                       model.pts_bbox_head,
                                       model.add_pred_to_datasample,
                                       model.video_test_mode)
    onnxModel = onnxModel.cpu()
    onnxModel.eval()

    return onnxModel


@master_only
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

    # Save input tensors
    """
    img_np                  = img.to('cpu').numpy()
    shift_xy_np             = shift_xy.to('cpu').numpy()
    rotation_grid_np        = rotation_grid.to('cpu').numpy()
    reference_points_cam_np = reference_points_cam.to('cpu').numpy()
    bev_mask_count_np       = bev_mask_count.to('cpu').numpy()
    bev_valid_indices_np    = bev_valid_indices.to('cpu').numpy()
    can_bus_np              = can_bus.to('cpu').numpy()
    prev_bev_np             = onnxModel.prev_frame_info['prev_bev'].to('cpu').numpy()

    np.savez('bevformer_input.npz', inputs=img_np, shift_xy=shift_xy_np,
             rotation_grid=rotation_grid_np, reference_points_cam=reference_points_cam_np,
             bev_mask_count=bev_mask_count_np, bev_valid_indices=bev_valid_indices_np,
             can_bus=can_bus_np, prev_bev=prev_bev_np)
    """

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
