#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch
import copy
import numpy as np

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES, YCBV_CLASSES, LM_CLASSES
from yolox.data import CADModelsYCBV, CADModelsLM
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, postprocess_object_pose, vis
from yolox.utils.object_pose_utils  import decode_rotation_translation
from yolox.utils.plots import plot_one_box, colors

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
_NUM_CLASSES = {"coco":80, "lm":15, "lmo": 8, "ycbv": 21, "coco_kpts":1}


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument( "--dataset", default=None, type=str, help="dataset is required for object_pose or human_pose estimation")
    parser.add_argument( "--task", default="2dod", type=str, help="dataset is required for object_pose or human_pose estimation")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
        task="2dod",
        data_set="coco"
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.task = task
        self.data_set = data_set
        self.cad_models = model.head.cad_models
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img_info["img"] = img.transpose(1, 2, 0)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            if self.task == "object_pose":
                outputs = postprocess_object_pose(
                    outputs, self.num_classes, self.confthre,
                    self.nmsthre, class_agnostic=True
                )
            else:
                outputs = postprocess(
                    outputs, self.num_classes, self.confthre,
                    self.nmsthre, class_agnostic=True
                )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

    def visual_object_pose(self, output, img_info, cls_conf):
        img = np.ascontiguousarray(img_info["img"])
        img_2d = copy.deepcopy(img)
        im_cuboid = copy.deepcopy(img)
        im_mask = copy.deepcopy(img)
        camera_matrix = self.cad_models.camera_matrix
        if output is None:
            return img
        output = output.cpu()
        if isinstance(camera_matrix, dict):
            camera_matrix = camera_matrix['camera_uw']
        for ind in range(output.shape[0]):
            pose = {}
            pose['xy'] = output[ind, 11:13]
            rotation_vec, translation_vec = decode_rotation_translation(output[ind], camera_matrix=camera_matrix)
            pose["rotation_vec"] = rotation_vec
            pose["translation_vec"] = translation_vec
            cls = output[ind][-1]
            color = colors(cls)
            plot_one_box(output[ind], img_2d, im_cuboid=im_cuboid, im_mask=im_mask, color=color, object_pose=True, label=str(int(cls.numpy())),
                  cad_models=self.cad_models, camera_matrix=camera_matrix, pose=pose, block_x=0, block_y=0, cls_names=self.cls_names)
        return [img, img_2d, im_cuboid, im_mask]



def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        if predictor.task == "object_pose":
            result_image = predictor.visual_object_pose(outputs[0], img_info, predictor.confthre)
        else:
            result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            if isinstance(result_image, list):
                images_type = ["raw_img", "box", "cuboid", "mask"]
                for image, image_type  in zip(result_image, images_type):
                    os.makedirs(os.path.join(save_folder, image_type), exist_ok=True)
                    save_file_name = os.path.join(save_folder, image_type, os.path.basename(image_name))
                    logger.info("Saving detection result in {}".format(save_file_name))
                    cv2.imwrite(save_file_name, image)
            else:
                save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                logger.info("Saving detection result in {}".format(save_file_name))
                cv2.imwrite(save_file_name, result_image)
            save_txt = True
            if save_txt:
                os.makedirs(os.path.join(save_folder, 'txt'), exist_ok=True)
                save_file_name = os.path.join(save_folder, 'txt', os.path.basename(image_name).split('.')[0]+'.txt')
                with open(save_file_name, 'a') as f:
                    for output in outputs[0]:
                        line = output.tolist()
                        f.write(('%8.5g  ' * len(line)).rstrip() % tuple(line) + '\n')

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
    if exp.data_set == "ycbv":
        cls_names = YCBV_CLASSES
    elif exp.data_set == "lmo" or exp.data_set== "lm":
        cls_names = LM_CLASSES
    else:
        cls_names = COCO_CLASSES

    predictor = Predictor(
        model, exp, cls_names, trt_file, decoder,
        args.device, args.fp16, args.legacy, args.task, exp.data_set
    )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.data_set = args.dataset if args.dataset is not None else exp.data_set
    exp.num_classes = _NUM_CLASSES[args.dataset]
    main(exp, args)
