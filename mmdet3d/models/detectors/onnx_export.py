import torch
import copy

from onnxsim import simplify
import onnx

from .onnx_network import PETR_export_model, DETR3D_export_model, \
                          BEVFormer_export_model, \
                          BEVDet_export_model, FCOS3D_export_model, \
                          FastBEV_export_model


def export_PETR(model, inputs=None, data_samples=None, **kwargs):

        onnxModel = PETR_export_model(model.img_backbone,
                                      model.img_neck,
                                      model.pts_bbox_head)
        onnxModel.eval()

        # Should clone. Otherwise, when we run both export_model and self.predict,
        # we have error PETRHead forward() - Don't know why
        img = inputs['imgs'].clone()

        batch_img_metas = [ds.metainfo for ds in data_samples]
        batch_img_metas = onnxModel.add_lidar2img(img, batch_img_metas)
        onnxModel.prepare_data(img, batch_img_metas)
        masks, coords3d = onnxModel.create_coords3d(img)

        # save masks and lidar_coor_1d
        masks_np    = masks.to('cpu').numpy()
        coords3d_np = coords3d.to('cpu').numpy()
        masks_np.tofile('petrv1_masks.dat')
        coords3d_np.tofile('petrv1_coords3d.dat')

        modelInput = []
        modelInput.append(img)
        modelInput.append(coords3d)

        # Passed the squeezed img
        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)

        torch.onnx.export(onnxModel,
                          tuple(modelInput),
                         'petrv1.onnx',
                          opset_version=16,
                          verbose=False)

        onnx_model, _ = simplify('petrv1.onnx')
        onnx.save(onnx_model, 'petrv1.onnx')

        print("!! ONNX model has been exported for PETR!!!\n\n")


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

    # Batch image to multiple branches
    N, C, H, W = img[0].shape
    img_list = [img[0][0].view(1, C, H, W),
                img[0][1].view(1, C, H, W),
                img[0][2].view(1, C, H, W),
                img[0][3].view(1, C, H, W),
                img[0][4].view(1, C, H, W),
                img[0][5].view(1, C, H, W)]

    modelInput = []
    modelInput.append(img_list)
    modelInput.append(bev_feat)
    modelInput.append(lidar_coor_1d)

    torch.onnx.export(onnxModel,
                      tuple(modelInput),
                     'bevDet.onnx',
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

        #onnxModel = BEVFormer_export_model(model.img_backbone,
        #                                   model.img_neck,
        #                                   model.pts_bbox_head,
        #                                   model.add_pred_to_datasample,
        #                                   model.video_test_mode)
        #onnxModel = onnxModel.cpu()
        #onnxModel.eval()

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

        print("!!ONNX model has been exported for BEVFormer!\n\n")


def export_FCOS3D(model, inputs=None, data_samples=None):

    onnxModel = FCOS3D_export_model(model.backbone,
                                    model.neck,
                                    model.bbox_head,
                                    model.add_pred_to_datasample)
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

    modelInput = []
    modelInput.append(img)
    modelInput.append(pad_cam2img)
    modelInput.append(inv_pad_cam2img)

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
                     'fcos3d.onnx',
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=16,
                      verbose=False)

    onnx_model, _ = simplify('fcos3d.onnx')
    onnx.save(onnx_model, 'fcos3d.onnx')

    print("!! ONNX model has been exported for FCOS3D! !!\n\n")



def create_onnx_FastBEV(model):
    onnxModel = FastBEV_export_model(model)
    onnxModel.eval()

    return onnxModel


def export_FastBEV(onnxModel, inputs=None,  data_samples=None):

    img = inputs['imgs'].clone()
    batch_img_metas = [ds.metainfo for ds in data_samples]

    onnxModel.prepare_data(batch_img_metas)

    prev_feats_map = None
    prev_input_metas = None

    if onnxModel.num_temporal_feats > 0:
        prev_feats_map, prev_input_metas = onnxModel.get_temporal_feats(
            onnxModel.queue, onnxModel.memory, img, batch_img_metas,
            onnxModel.feats_size, onnxModel.num_temporal_feats)

    xy_coors = onnxModel.precompute_proj_info_for_inference(img, batch_img_metas, prev_img_metas=prev_input_metas)

    model_input = []
    model_input.append(img)
    model_input.append(xy_coors)
    model_input.append(prev_feats_map)

    input_names  = ["imgs", "xy_coor", "prev_img_feats"]
    output_names = ["img_feats", "bboxes", "scores", "dir_scores"]

    torch.onnx.export(onnxModel,
                      tuple(model_input),
                      'fastBEV.onnx',
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=16,
                      verbose=False)

    onnx_model, _ = simplify('fastBEV.onnx')
    onnx.save(onnx_model, 'fastBEV.onnx')

    print("\n!! ONNX model has been exported for FastBEV!!!\n\n")

