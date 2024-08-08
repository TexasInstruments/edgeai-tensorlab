import torch
import copy

from .onnx_network import PETR_export_model, DETR3D_export_model, BEVFormer_export_model, BEVDet_export_model


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

    #batch_img_metas = onnxModel.add_lidar2img(img, batch_img_metas)
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


def export_BEVFormer(model, inputs=None, data_samples=None, **kwargs):

        onnxModel = BEVFormer_export_model(model.img_backbone, 
                                           model.img_neck,
                                           model.pts_bbox_head,
                                           model.add_pred_to_datasample,
                                           model.video_test_mode)
        onnxModel = onnxModel.cpu()
        onnxModel.eval()

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

        # copy batch_img_metas
        onnxModel.prepare_data(img, batch_img_metas)

        modelInput = []
        modelInput.append(img)
        modelInput.append(onnxModel.prev_frame_info['prev_bev'])

        torch.onnx.export(onnxModel,
                          tuple(modelInput),
                         'bevFormer.onnx',
                          opset_version=16,
                          verbose=False)

        onnxModel.prev_frame_info['prev_pos']   = tmp_pos
        onnxModel.prev_frame_info['prev_angle'] = tmp_angle
        #onnxModel.prev_frame_info['prev_bev']   = outs['bev_embed']

        print("!!ONNX model has been exported for BEVFormer!\n\n")

