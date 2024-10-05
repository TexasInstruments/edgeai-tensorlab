#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os, copy

import cv2
import numpy as np

import onnxruntime as rt
from tqdm import tqdm

from object_pose_utils_onnx import get_cuboid_corner, get_camera_matrix, get_class_names, draw_obj_pose, draw_bbox_2d


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image-folder",
        type=str,
        default='./test_image/',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='ycbv',
        help="dataset type",
    )
    parser.add_argument(
        "-s",
        "--score-thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default="640,480",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Whether to write the output ",
    )
    parser.add_argument(
        "--tidl-delegate",
         action="store_true",
         help="use tidl_delegate"
    )
    parser.add_argument(
        "--compile",
         action="store_true",
         help="use tidl_delegate and compile"
    )

    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    input_shape = tuple(map(int, args.input_shape.split(',')))
    image_file_list = sorted(os.listdir(args.image_folder))
    pbar = tqdm(enumerate(image_file_list), total=len(image_file_list))
    prototxt = args.model.replace("onnx", "prototxt")
    assert os.path.exists(prototxt), "Prototxt not available. Please provide a prototxt {}".format(prototxt)
    #session = rt.InferenceSession(args.model)
    so = rt.SessionOptions()
    if args.tidl_delegate:
        compile_options = {
            "artifacts_folder": "./artifacts",
            "tensor_bits": 16,
            "accuracy_level": 1,
            # "debug_level": 3,
            "advanced_options:calibration_frames": 25,
            "advanced_options:calibration_iterations": 2,
            # "advanced_options:output_feature_16bit_names_list" : "370, 680, 990, 1300",
            'object_detection:meta_layers_names_list': prototxt,
            'object_detection:meta_arch_type': 6,
            "ti_internal_nc_flag": 1601,
            # "add_data_convert_ops" : 3,
        }
        if args.compile:
            EP_list = ['TIDLCompilationProvider','CPUExecutionProvider']
            compile_options["tidl_tools_path"] = os.environ["TIDL_TOOLS_PATH"]
            os.makedirs(compile_options["artifacts_folder"], exist_ok=True)
            for root, dirs, files in os.walk(compile_options["artifacts_folder"], topdown=False):
                [os.remove(os.path.join(root, f)) for f in files]
                [os.rmdir(os.path.join(root, d)) for d in dirs]
        else:
            EP_list = ['TIDLExecutionProvider','CPUExecutionProvider']
            compile_options["tidl_tools_path"] = ""
        session = rt.InferenceSession(args.model ,providers=EP_list, provider_options=[compile_options, {}], sess_options=so)
    else:
        compile_options = {}
        EP_list = ['CPUExecutionProvider']
        session = rt.InferenceSession(args.model ,providers=EP_list, provider_options=[compile_options], sess_options=so)

    for img_index, image_file in pbar:
        if image_file.endswith("png"):
            image_path = os.path.join(args.image_folder, image_file)
            origin_img = cv2.imread(image_path)
            origin_img = cv2.resize(origin_img, input_shape, interpolation=cv2.INTER_LINEAR)
            img = origin_img.transpose((2, 0, 1))
            img = np.ascontiguousarray(img, dtype=np.float32)
            ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
            output = session.run(None, ort_inputs)
            if args.compile:
                if img_index >= compile_options["advanced_options:calibration_frames"]-1:
                    break
            else:
                class_to_cuboid = get_cuboid_corner(dataset=args.dataset)
                camera_matrix = get_camera_matrix(dataset=args.dataset)
                class_names = get_class_names(dataset=args.dataset)
                dets = output[0]
                img_2d_od = copy.deepcopy(origin_img)

                if dets is not None:
                    draw_bbox_2d(img_2d_od, dets, class_names)
                    draw_obj_pose(origin_img, dets, class_names=class_names, class_to_cuboid=class_to_cuboid, camera_matrix=camera_matrix)
                os.makedirs(args.output_dir, exist_ok=True)
                output_path_cuboid = os.path.join(args.output_dir, image_path.split("/")[-1])
                cv2.imwrite(output_path_cuboid, origin_img)
                os.makedirs(os.path.join(args.output_dir, 'bbox'), exist_ok=True)
                output_path_2d_od = os.path.join(args.output_dir, 'bbox', image_path.split('/')[-1])
                cv2.imwrite(output_path_2d_od, img_2d_od)

                if args.save_txt:  # Write to file in tidl dump format
                    if len(dets.shape) > 2:
                        dets = dets[0][0]
                    output_txt_path = os.path.join(os.path.dirname(output_path_cuboid) , os.path.basename(output_path_cuboid).split('.')[0] + '.txt')
                    sort_index = np.argsort(dets[:,5])
                    dets = dets[sort_index]
                    for det in dets:
                        box, score, cls = det[:4], det[4], int(det[5])
                        r1, r2 = det[6:9], det[9:12]
                        t = det[12:15]
                        line = (cls, score, *box, *r1, *r2, *t)
                        with open(output_txt_path, 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
