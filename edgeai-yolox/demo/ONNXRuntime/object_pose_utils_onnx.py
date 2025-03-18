#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import numpy as np
import cv2

#camera_matrix for test-split of ycbv dataset
ycbv_camera_matrix = np.array([[1066.778, 0, 312.9869],
                              [0.0, 1067.487, 241.3109],
                              [0.0, 0.0, 1.0]], dtype=np.float32)

#camera_matrix for linemod dataset
lm_camera_matrix = np.array([[572.4114, 0.0, 325.2611],
                             [0.0, 573.57043, 242.04899],
                             [ 0.0, 0.0, 1.0]], dtype=np.float32)

# camera_matrix for logitech c270 ->1280x960
logitech_camera_matrix_raw = np.array([[1430, 0.0, 620],
                                        [0.0, 1430, 480],
                                        [ 0.0, 0.0, 1.0]], dtype=np.float32)

# camera_matrix for logitech c270 ->640x480
logitech_camera_matrix_resize = np.array([[1066.778, 0.0, 312.9869],
                                          [0.0, 1066.778, 241.3109],
                                          [0.0, 0.0, 1.0]], dtype=np.float32)

#vertices for ycbv21 objects
ycbv_vertices = np.array([
    [51.1445, 51.223, 70.072],
    [35.865, 81.9885, 106.743],
    [24.772, 47.024, 88.0075],
    [33.927, 33.875, 51.0185],
    [48.575, 33.31, 95.704],
    [42.755, 42.807, 16.7555],
    [68.924, 64.3955, 19.414],
    [44.6775, 50.5545, 15.06],
    [51.0615, 30.161, 41.8185],
    [54.444, 89.206, 18.335],
    [74.4985, 72.3845, 121.32],
    [51.203, 33.856, 125.32],
    [80.722, 80.5565, 27.485],
    [58.483, 46.5375, 40.692],
    [92.1205, 93.717, 28.6585],
    [51.9755, 51.774, 102.945],
    [48.04, 100.772, 7.858],
    [10.5195, 60.4225, 9.4385],
    [59.978, 85.639, 19.575],
    [104.897, 82.18, 18.1665],
    [26.315, 38.921, 25.5655,]
    ], dtype=np.float32)

#vertices for Linemod objects
lm_vertices = np.array([
    [-37.93430000,  38.79960000,  45.88450000],
    [107.83500000,  60.92790000,  109.70500000],
    [83.21620000,  82.65910000,  37.23640000],
    [68.32970000,  71.51510000,  50.24850000],
    [50.39580000,  90.89790000,  96.86700000],
    [33.50540000,  63.81650000,  58.72830000],
    [58.78990000,  45.75560000,  47.31120000],
    [114.73800000,  37.73570000,  104.00100000],
    [52.21460000,  38.70380000,  42.84850000],
    [75.09230000,  53.53750000,  34.62070000],
    [18.36050000,  38.93300000,  86.40790000],
    [50.44390000,  54.24850000,  45.40000000],
    [129.11300000,  59.24100000,  70.56620000],
    [101.57300000,  58.87630000,  106.55800000],
    [46.95910000,  73.71670000,  92.37370000]
    ], dtype=np.float32)

#order of vertices to draw cuboid
vertices_order = np.array([
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1,  1],
            [-1, 1, -1],
            [1, -1, -1],
            [1, -1,  1],
            [1,  1,  1],
            [1,  1, -1],
            ], dtype=np.float32)

YCBV_CLASSES = (
     "master_chef_can",
     "cracker_box",
     "sugar_box",
     "tomato_soup_can",
     "mustard_bottle",
     "tuna_fish_can",
     "pudding_box",
     "gelatin_box",
     "potted_meat_can",
     "banana",
     "pitcher_base",
     "bleach_cleanser",
     "bowl",
     "mug",
     "power_drill",
     "wood_block",
     "scissors",
     "large_marker",
     "large_clamp",
     "extra_large_clamp",
     "foam_brick"
)

LM_CLASSES = (
     "ape",
     "benchvise",
     "bowl",
     "cam",
     "can",
     "cat",
     "cup",
     "driller",
     "duck",
     "eggbox",
     "glue",
     "holepuncher",
     "iron",
     "lamp",
     "phone"
)

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def get_cuboid_corner(dataset="ycbv"):
    if dataset == "ycbv":
        return ycbv_vertices[:, None, :] * vertices_order
    else:
        return lm_vertices[:, None, :] * vertices_order


def get_camera_matrix(dataset="ycbv"):
    if dataset == "ycbv":
        return ycbv_camera_matrix
    else:
        return lm_camera_matrix

def get_class_names(dataset="ycbv"):
    if dataset == "ycbv":
        return YCBV_CLASSES
    else:
        return LM_CLASSES

def draw_cuboid_2d(img, cuboid_corners, color = (0, 255, 0), thickness = 2):
    box = np.copy(cuboid_corners).astype(np.int32)
    box = [tuple(kpt) for kpt in box]
    #front??? to check
    cv2.line(img, box[0], box[1], color, thickness)
    cv2.line(img, box[1], box[2], color, thickness)
    cv2.line(img, box[2], box[3], color, thickness)
    cv2.line(img, box[0], box[3], color, thickness)
    #back
    cv2.line(img, box[4], box[5], color, thickness)
    cv2.line(img, box[5], box[6], color, thickness)
    cv2.line(img, box[6], box[7], color, thickness)
    cv2.line(img, box[4], box[7], color, thickness)
    #sides
    cv2.line(img, box[0], box[4], color, thickness)
    cv2.line(img, box[1], box[5], color, thickness)
    cv2.line(img, box[2], box[6], color, thickness)
    cv2.line(img, box[3], box[7], color, thickness)
    return img


def draw_bbox_2d(origin_img, dets, class_names, conf_thres=0.85):
    if len(dets.shape) > 2:
        dets = dets[0][0]

    for det in dets:
        box, score, cls = det[:4], det[4], int(det[5])
        if score>conf_thres:
            cls_id = int(cls)
            x0, y0 = int(box[0]), int(box[1])
            x1, y1 = int(box[2]), int(box[3])
            color = (_COLORS[cls] * 255).astype(np.uint8).tolist()
            cv2.rectangle(origin_img, (x0, y0), (x1, y1), color, 2)
                    #Labels on cuboid
            text = '{}:{:.1f}%'.format(class_names[cls], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            txt_bk_color = (_COLORS[cls] * 255 * 0.7).astype(np.uint8).tolist()
            x0, y0 = int(box[0]), int(box[1])
            cv2.rectangle(
                origin_img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(origin_img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
    return origin_img


def project_3d_2d(pts_3d, rotation_mat, translation_vec, camera_matrix):
    xformed_3d = np.matmul(pts_3d, rotation_mat.T) + translation_vec
    xformed_3d[:,:3] = xformed_3d[:,:3]/xformed_3d[:,2:3]
    projected_2d = np.matmul(xformed_3d, camera_matrix.reshape((3, 3)).T)[:, :2]
    return projected_2d


def draw_obj_pose(origin_img, dets, class_names, class_to_cuboid, camera_matrix, conf_thres=0.85):
    if len(dets.shape) > 2:
        dets = dets[0][0]

    for det in dets:
        box, score, cls = det[:4], det[4], int(det[5])
        if score>conf_thres:
            color = (_COLORS[cls] * 255).astype(np.uint8).tolist()
            r1, r2 = det[6:9, None], det[9:12, None]
            r3 = np.cross(r1, r2, axis=0)
            rotation_mat = np.concatenate((r1, r2, r3), axis=1)
            translation_vec = det[12:15]
            tx = translation_vec[0]
            ty = translation_vec[1]
            tz = translation_vec[2]
            x = camera_matrix[0,2] + camera_matrix[0,0] *tx/tz
            y = camera_matrix[1,2] + camera_matrix[1,1] *ty/tz
            X = (x - logitech_camera_matrix_resize[0,2])*tz/logitech_camera_matrix_resize[0,0]
            Y = (y - logitech_camera_matrix_resize[1,2])*tz/logitech_camera_matrix_resize[1,1]
            translation_vec[0] = X
            translation_vec[1] = Y
            cuboid_corners_2d = project_3d_2d(class_to_cuboid[int(cls)], rotation_mat, translation_vec, camera_matrix)
            draw_cuboid_2d(img=origin_img, cuboid_corners=cuboid_corners_2d, color=color)

            #Labels on cuboid
            text = '{}:{:.1f}%'.format(class_names[cls], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            txt_bk_color = (_COLORS[cls] * 255 * 0.7).astype(np.uint8).tolist()
            x0, y0 = int(box[0]), int(box[1])
            cv2.rectangle(
                origin_img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(origin_img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
    return origin_img
