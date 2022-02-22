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
    show = False
    out_dir = './show-dir'
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    dump_txt_op = False
    read_txt_op = True
    for i, data in enumerate(data_loader):
        #if i >=100:
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

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
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
    return results
