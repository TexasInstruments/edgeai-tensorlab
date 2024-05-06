# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import mmcv
import torch
from mmcv.image import tensor2imgs

from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): Whether to save viualization results.
            Default: True.
        out_dir (str, optional): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    dump_txt_op = False
    read_txt_op = False
    en_img_draw = False
    dump_bin_op = False
    for i, data in enumerate(data_loader):
        #if i >=200:
        #    break
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if dump_txt_op:
            img_metas = data['img_metas'][0].data[0][0]
            file_name = osp.split(img_metas['pts_filename'])[1]
            f = open(file_name+'.txt','w')
            for score, label, det_tensor in zip(result[0]['scores_3d'], result[0]['labels_3d'],result[0]['boxes_3d'].tensor):
                f.write("{:3d} ".format(label))
                f.write("{:.4f} ".format(score))
                f.write("{:.4f} {:.4f} {:.4f} ".format(det_tensor[0],det_tensor[1],det_tensor[2]))
                f.write("{:.4f} {:.4f} {:.4f} ".format(det_tensor[3],det_tensor[4],det_tensor[5]))
                f.write("{:.4f}".format(det_tensor[6]))
                f.write("\n")
            f.close()

        if read_txt_op:
            img_metas = data['img_metas'][0].data[0][0]
            file_name = osp.split(img_metas['pts_filename'])[1]
            file_name = osp.join('/user/a0393749/deepak_files/ti/c7x-mma-tidl-before/ti_dl/test/testvecs/output',file_name)
            f = open(file_name+'.txt','r')
            lines = f.readlines()
            det_tensor = torch.empty((len(lines),7), dtype=torch.float32, device = 'cpu')

            result[0]['scores_3d'] = torch.empty((len(lines)), dtype=torch.float32, device = 'cpu')
            result[0]['labels_3d'] = torch.empty((len(lines)), dtype=torch.float32, device = 'cpu')
            result[0]['boxes_3d']  = img_metas['box_type_3d'](det_tensor, box_dim=7)

            for i, line in enumerate(lines):
                det = line.strip().split()
                result[0]['labels_3d'][i] = float(det[0])
                result[0]['scores_3d'][i] = float(det[1])
                for j in range(7):
                    det_tensor[i][j] = float(det[j+2])
            result[0]['boxes_3d'] = img_metas['box_type_3d'](det_tensor, box_dim=7)

            f.close()

        if en_img_draw:
            import cv2
            import numpy as np

            from mmdet3d.core.visualizer import image_vis

            ip_img_folder = 'data/kitti/training/image_2'
            op_img_folder = '/user/a0393749/deepak_files/temp/output_platform'
            vis_score_th  = 0.5
            color=(255, 255, 0)
            
            img_metas = data['img_metas'][0].data[0][0]
            file_name = osp.split(img_metas['pts_filename'])[1].replace('.bin','.png')
            
            ip_file_name = osp.join(ip_img_folder,file_name)
            op_file_name = osp.join(op_img_folder,file_name)
            
            img = cv2.imread(ip_file_name)
            valid_ids = result[0]['scores_3d'] > vis_score_th
            boxes3d = result[0]['boxes_3d']

            corners_3d = boxes3d.corners
            num_bbox = corners_3d.shape[0]
            pts_4d = np.concatenate(
                [corners_3d.reshape(-1, 3),
                np.ones((num_bbox * 8, 1))], axis=-1)
            lidar2img_rt = img_metas['lidar2img'].reshape(4, 4)
            if isinstance(lidar2img_rt, torch.Tensor):
                lidar2img_rt = lidar2img_rt.cpu().numpy()
            pts_2d = pts_4d @ lidar2img_rt.T

            pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
            pts_2d[:, 0] /= pts_2d[:, 2]
            pts_2d[:, 1] /= pts_2d[:, 2]
            imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)
            
            for obj_id in range(num_bbox):
                if result[0]['scores_3d'][obj_id] > vis_score_th:
                    if result[0]['labels_3d'][obj_id] == 0:
                        color = (0,255,255)
                    elif result[0]['labels_3d'][obj_id] == 1:
                        color = (0,255,0)
                    else:
                        color = (0,0,255)
                    image_vis.plot_rect3d_on_img(img, 1, imgfov_pts_2d[[obj_id],:], color, 1)

            cv2.imwrite(op_file_name, img)

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d) and en_img_draw == False:
                model.module.show_results(
                    data,
                    result,
                    out_dir=out_dir,
                    show=show,
                    score_thr=show_score_thr)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    if dump_bin_op == True:
        bin_out = []
        import numpy as np
        for result in results:
            tidl_format_op = np.concatenate((np.expand_dims(result['labels_3d'].cpu().numpy(),axis=1),
            np.expand_dims(result['scores_3d'].cpu().numpy(),axis=1),
            result['boxes_3d'].tensor.cpu().numpy()), axis=1)
            bin_out.append(tidl_format_op)

        np.save('bin_out.bin', bin_out)

    return results
