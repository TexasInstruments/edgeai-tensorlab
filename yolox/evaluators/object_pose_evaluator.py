#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
import os
from loguru import logger
from tqdm import tqdm
import cv2
from ..utils  import visualize_object_pose, decode_rotation_translation
import numpy as np
from sklearn.neighbors import KDTree
import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    postprocess_object_pose,
    synchronize,
    time_synchronized,
    xyxy2xywh
)



class ObjectPoseEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, dataloader, img_size, confthre, nmsthre, num_classes, testdev=False, visualize = False, output_dir=None
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.visualize = visualize
        self.output_dir = output_dir
        self.class_to_model = dataloader.dataset.class_to_model
        self.class_to_cuboid = dataloader.dataset.models_corners

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        pred_data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, targets, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                predicted_pose = postprocess_object_pose(outputs, self.num_classes, self.confthre, self.nmsthre)

                frame_data_list, frame_pred_data_list = self.convert_to_coco_format(predicted_pose, targets, info_imgs, ids)
                data_list.extend(frame_data_list)
                pred_data_list.extend(frame_pred_data_list)

                if self.visualize:
                    os.makedirs(os.path.join(self.output_dir, "vis_pose"), exist_ok=True)
                    for output_idx in range(len(predicted_pose)):
                        img = imgs[output_idx]
                        visualize_object_pose.draw_6d_pose(img, frame_data_list, class_to_model=self.class_to_model,
                                                                    class_to_cuboid=self.class_to_cuboid, out_dir=self.output_dir, id=ids[output_idx][0])
                        visualize_object_pose.draw_6d_pose(img, frame_data_list, class_to_model=self.class_to_model,
                                                                    class_to_cuboid=self.class_to_cuboid, gt=False, out_dir=self.output_dir,id=ids[output_idx][0])
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end



        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            pred_data_list = gather(pred_data_list, dst=0)
            pred_data_list = list(itertools.chain(*pred_data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(pred_data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, targets, info_imgs, ids):
        data_list = []
        pred_list = []
        for (output, target, img_h, img_w, img_id) in zip(
            outputs, targets, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output, target = output.cpu(), target.cpu()
            bboxes = output[:, 0:4]
            bboxes_gt = target[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes_gt /= scale
            bboxes = xyxy2xywh(bboxes)
            bboxes_gt = xyxy2xywh(bboxes_gt)

            cls = target[:, 4]
            cls_pred = output[:, -1]
            scores = output[:, 4] * output[:, -2]
            for ind in range(bboxes.shape[0]):
                pred_label = self.dataloader.dataset.class_ids[int(cls_pred[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": pred_label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                pred_list.append(pred_data)

            for ind in range(bboxes_gt.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                if len(output[output[:, -1] == label]) ==1 :
                    missing_det = False
                    rotation_pred, translation_pred = decode_rotation_translation(output[output[:, -1] == label][0])
                else:
                    missing_det=True
                rotation_gt, translation_gt = decode_rotation_translation(target[ind])
                pred_gt_data = {
                    "image_id": int(img_id),
                    "bbox_gt": bboxes_gt[ind].numpy().tolist(),
                    "rotation_gt": rotation_gt.tolist(),
                    "translation_gt": translation_gt.tolist(),
                    "xy_gt": target[ind][11:13].numpy().tolist(),
                    "category_id": label,
                    "missing_det": True
                }
                if not missing_det:
                    pred_gt_data.update(
                    {
                        "image_id": int(img_id),
                        "category_id": label,
                        "bbox": bboxes[output[:, -1]==label][0].numpy().tolist(),
                        "bbox_pred": bboxes[output[:, -1] == label][0].numpy().tolist(),
                        "score": scores[output[:, -1]==label].numpy().item(),
                        "segmentation": [],
                        "rotation_pred" : rotation_pred.tolist(),
                        "translation_pred": translation_pred.tolist(),
                        "xy_pred": output[output[:, -1] == label][0][11:13].numpy().tolist(),
                        "missing_det": False
                    }) # COCO json format

                data_list.append(pred_gt_data)
        return data_list, pred_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info


    def evaluate_prediction_6dpose(self, data_dict, statistics):
        if args.object_name in ['eggbox', 'glue']:
            compute_score = compute_adds_score
        else:
            compute_score = compute_add_score

        score = compute_score(pts3d,
                                   diameter,
                                   (record['R_gt'], record['t_gt']),
                                   (record['R_init'], record['t_init']))
        print('ADD(-S) score of initial prediction is: {}'.format(score))



    def compute_add_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):
        R_gt, t_gt = pose_gt
        R_pred, t_pred = pose_pred
        count = R_gt.shape[0]
        mean_distances = np.zeros((count,), dtype=np.float32)
        for i in range(count):
            pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]
            pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
            distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
            mean_distances[i] = np.mean(distance)

        threshold = diameter * percentage
        score = (mean_distances < threshold).sum() / count
        return score



    def compute_adds_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):
        R_gt, t_gt = pose_gt
        R_pred, t_pred = pose_pred

        count = R_gt.shape[0]
        mean_distances = np.zeros((count,), dtype=np.float32)
        for i in range(count):
            if np.isnan(np.sum(t_pred[i])):
                mean_distances[i] = np.inf
                continue
            pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]
            pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
            kdt = KDTree(pts_xformed_gt.transpose(), metric='euclidean')
            distance, _ = kdt.query(pts_xformed_pred.transpose(), k=1)
            mean_distances[i] = np.mean(distance)
        threshold = diameter * percentage
        score = (mean_distances < threshold).sum() / count
        return score
