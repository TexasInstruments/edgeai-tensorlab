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
from ..utils  import visualize_object_pose
from ..utils.object_pose_utils  import decode_rotation_translation
from ..data.datasets.lmo import LMODataset
from ..data.datasets.ycbv import YCBVDataset
import numpy as np
from sklearn.neighbors import KDTree
import torch
import csv

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
        self, dataloader, img_size, confthre, nmsthre, num_classes, testdev=False, visualize = False, output_dir=None, test_bop=True
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
        self.test_bop = test_bop  #dumps the output in BOP format
        self.dataset = None
        if self.test_bop:
            if isinstance(dataloader.dataset, YCBVDataset):
                self.dataset = 'ycbv'
                self.class_map = None
            elif isinstance(dataloader.dataset, LMODataset):
                self.dataset = 'lmo'
                self.class_map = {v:k for k,v in dataloader.dataset.class_map.items()}
            else:
                assert "Provided dataset is not supported for 6d pose evaluation"
        self.test_bop_file_name = "bop_{}-test.csv".format(self.dataset)
        self.output_dir = output_dir
        self.class_to_model = dataloader.dataset.class_to_model
        self.class_to_cuboid = dataloader.dataset.models_corners
        self.class_to_name = dataloader.dataset.class_to_name
        self.class_to_diameter = dataloader.dataset.models_diameter
        self.symmetric_objects = dataloader.dataset.symmetric_objects

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
                    import time
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                predicted_pose = postprocess_object_pose(outputs, self.num_classes, self.confthre, self.nmsthre)
                if isinstance(self.dataloader.dataset.cad_models.camera_matrix, dict):
                    camera_matrix = self.dataloader.dataset.cad_models.camera_matrix['camera_uw']
                else:
                    camera_matrix = self.dataloader.dataset.cad_models.camera_matrix
                frame_data_list, frame_pred_data_list = self.convert_to_coco_format(predicted_pose, targets, info_imgs, ids, camera_matrix)
                data_list.extend(frame_data_list)
                pred_data_list.extend(frame_pred_data_list)

                if self.visualize:
                    os.makedirs(os.path.join(self.output_dir, "vis_pose"), exist_ok=True)
                    for output_idx in range(len(predicted_pose)):
                        img = imgs[output_idx]
                        if len(frame_data_list) != 0:
                            img_cuboid, img_mask, img_2dod = visualize_object_pose.draw_6d_pose(img, frame_data_list, camera_matrix, class_to_model=self.class_to_model,
                                                                        class_to_cuboid=self.class_to_cuboid, out_dir=self.output_dir, id=ids[output_idx])
                            visualize_object_pose.draw_6d_pose(img, frame_data_list, camera_matrix, class_to_model=self.class_to_model,
                                                                    class_to_cuboid=self.class_to_cuboid, gt=False, out_dir=self.output_dir, id=ids[output_idx],
                                                                    img_cuboid=img_cuboid, img_mask=img_mask, img_2dod=img_2dod)
                    if self.test_bop and None not in predicted_pose:
                        csv_file_path = os.path.join(self.output_dir, self.test_bop_file_name)
                        with open(csv_file_path, 'a') as csv_file:
                            csvwriter = csv.writer(csv_file)
                            for pose_per_object in predicted_pose[output_idx]:
                                id = int(ids[output_idx])
                                pose_per_object = pose_per_object.cpu().numpy()
                                scene_id = int(self.dataloader.dataset.coco.imgs[id]['image_folder'])
                                image_id = int(self.dataloader.dataset.coco.imgs[id]['file_name'].split('.')[0])
                                if self.class_map is not None:
                                    object_id = int(self.class_map[int(pose_per_object[-1])] + 1)
                                else:
                                    object_id = int(pose_per_object[-1] + 1)
                                score = pose_per_object[4]
                                r11, r21, r31, r12, r22, r32 = pose_per_object[5:11]
                                r13, r23, r33 = np.cross(pose_per_object[5:8], pose_per_object[8:11])
                                R = "{} {} {} {} {} {} {} {} {}".format(r11, r12, r13, r21, r22, r23, r31, r32, r33)
                                tx, ty = pose_per_object[11:13]
                                tz = pose_per_object[13] * 100.0
                                tx = (tx - camera_matrix[2]) * tz / camera_matrix[0]
                                ty = (ty - camera_matrix[5]) * tz / camera_matrix[4]
                                T = "{} {} {}".format(tx, ty, tz)
                                time = -1
                                bop_pose = [scene_id, image_id, object_id, score, R, T, time]
                                csvwriter.writerow(bop_pose)

                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            pred_data_list = gather(pred_data_list, dst=0)
            data_list = gather(data_list, dst=0)
            pred_data_list = list(itertools.chain(*pred_data_list))
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results_6dpose = self.evaluate_prediction_6dpose(data_list, statistics)
        eval_results_2d_od = self.evaluate_prediction_2d_od(pred_data_list, statistics, eval_results_6dpose)
        synchronize()
        return  eval_results_2d_od

    def convert_to_coco_format(self, outputs, targets, info_imgs, ids, camera_matrix):
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
                if len(output[output[:, -1] == label]) == 1 :
                    missing_det = False
                    rotation_pred, translation_pred = decode_rotation_translation(output[output[:, -1] == label][0], camera_matrix)
                else:
                    missing_det=True
                rotation_gt, translation_gt = decode_rotation_translation(target[ind], camera_matrix)
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

    def evaluate_prediction_2d_od(self, data_dict, statistics, pose_info):
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
            info += pose_info[-1]
            return pose_info[0]['ADD_0p1_avg'], cocoEval.stats[1], info
        else:
            return 0, 0, info


    def evaluate_prediction_6dpose(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0
        data_dict_asym = []
        data_dict_sym = []
        for pred_data in data_dict:
            if self.class_to_name[pred_data['category_id']] not in self.symmetric_objects.values():
                data_dict_asym.extend([pred_data])
            else:
                data_dict_sym.extend([pred_data])
        score_dict_asym = self.compute_add_score(data_dict_asym)
        score_dict_sym = self.compute_adds_score(data_dict_sym)
        score_dict = {}
        for metric in score_dict_asym.keys():
            score_dict[metric] = {**score_dict_asym[metric], **score_dict_sym[metric]}
            score_dict[metric + "_avg"] = np.mean(list(score_dict[metric].values()))

        score_dict_summmary = ""
        for metric in score_dict.keys():
            score_dict_summmary += metric + "\n"
            if "avg" not in metric:
                for cls in score_dict[metric].keys():
                    score_dict_summmary += "{:>5}".format(cls) + " ({:>12}) : ".format(self.class_to_name[cls]) + "{0:2f}".format(score_dict[metric][cls]) + "\n"
            else:
                score_dict_summmary += "{0:2f}".format(score_dict[metric]) + "\n"

        return score_dict, score_dict_summmary


    def compute_add_score(self, data_dict, percentage=0.1):
        distance_category = np.zeros((len(data_dict), 2))   #stores distance and category for each object in a row
        for index, pred_data in enumerate(data_dict):
            if not pred_data['missing_det']:
                R_gt, t_gt = np.array(pred_data['rotation_gt']), np.array(pred_data['translation_gt'])
                R_gt, _ = cv2.Rodrigues(R_gt)
                R_pred, t_pred= np.array(pred_data['rotation_pred']), np.array(pred_data['translation_pred'])
                R_pred, _ = cv2.Rodrigues(R_pred)
                pts3d = self.class_to_model[pred_data['category_id']]
                #mean_distances = np.zeros((count,), dtype=np.float32)
                pts_xformed_gt = np.matmul(R_gt,  pts3d.transpose()) + t_gt[:, None]
                pts_xformed_pred = np.matmul(R_pred, pts3d.transpose()) + t_pred[:, None]
                distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
                distance_category[index, 0] = np.mean(distance)
                distance_category[index, 1] = pred_data['category_id']
            else:
                distance_category[index, 0] = 1e6 # This distance must be set to a very high value for missign detections.
                distance_category[index, 1] = pred_data['category_id']

        threshold = [self.class_to_diameter[category] * percentage for category in self.class_to_diameter.keys()]
        score_dict = {}
        for threshold_multiple in range(1, 6):
            score_dict["ADD_0p{}".format(threshold_multiple)] = {}
        for category_id, category_type in self.class_to_name.items():
            num_instances = len(distance_category[distance_category[:, 1] == category_id][:,0])
            if num_instances > 0:
                for threshold_multiple in range(1,6):
                    score = np.sum(distance_category[distance_category[:, 1] == category_id][:,0] < threshold_multiple * threshold[category_id]) / (num_instances + 1e-6)
                    score_dict["ADD_0p{}".format(threshold_multiple)].update({category_id: score})
        return score_dict


    def compute_adds_score(self, data_dict, percentage=0.1):
        distance_category = np.zeros((len(data_dict), 2))
        for index, pred_data in enumerate(data_dict):
            if not pred_data['missing_det']:
                R_gt, t_gt = np.array(pred_data['rotation_gt']), np.array(pred_data['translation_gt'])
                R_gt, _ = cv2.Rodrigues(R_gt)
                R_pred, t_pred = np.array(pred_data['rotation_pred']), np.array(pred_data['translation_pred'])
                R_pred, _ = cv2.Rodrigues(R_pred)
                pts3d = self.class_to_model[pred_data['category_id']]
                pts_xformed_gt = np.matmul(R_gt, pts3d.transpose()) + t_gt[:, None]
                pts_xformed_pred = np.matmul(R_pred, pts3d.transpose()) + t_pred[:, None]
                kdt = KDTree(pts_xformed_gt.transpose(), metric='euclidean')
                distance, _ = kdt.query(pts_xformed_pred.transpose(), k=1)
                # distance_np = np.sqrt(     #brute-force distance calculation
                #     np.min(np.sum((pts_xformed_gt[:, :, None] - pts_xformed_pred[:, None, :]) ** 2, axis=0), axis=1))
                distance_category[index, 0] = np.mean(distance)
                distance_category[index, 1] = pred_data['category_id']
            else:
                distance_category[index, 0] = 1e6 # This must be set to a very high value
                distance_category[index, 1] = pred_data['category_id']


        threshold = [self.class_to_diameter[category] * percentage for category in self.class_to_diameter.keys()]
        score_dict = {}
        for threshold_multiple in range(1, 6):
            score_dict["ADD_0p{}".format(threshold_multiple)] = {}

        for category_id, category_type in self.class_to_name.items():
            num_instances = len(distance_category[distance_category[:, 1] == category_id][:, 0])
            if num_instances > 0:
                for threshold_multiple in range(1,6):
                    score = np.sum(distance_category[distance_category[:, 1] == category_id][:,0] < threshold_multiple * threshold[category_id]) / (num_instances + 1e-6)
                    score_dict["ADD_0p{}".format(threshold_multiple)].update({category_id: score})
        return score_dict
