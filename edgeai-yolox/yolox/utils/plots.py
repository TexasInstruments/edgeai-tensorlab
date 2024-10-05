# Plotting utils

import glob
import math
import os
import random
import copy
from pathlib import Path

import cv2

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

#from utils.general import xywh2xyxy, xyxy2xywh
from .boxes import cxcywh2xyxy, xyxy2cxcywh
from .visualize_object_pose import project_3d_2d, draw_cuboid_2d, draw_bbox_2d, Colors
from .object_pose_utils import decode_rotation_translation
from ..data.datasets.ycbv import YCBVDataset

colors = Colors()  # create instance for 'from utils.plots import colors'


def plot_one_box(x, im, im_cuboid=None, im_mask=None, color=None, label=None, line_thickness=3, human_pose=False, object_pose=False, kpts=None, steps=2, orig_shape=None, pose=None,
                 cad_models=None, camera_matrix=None, block_x=None, block_y=None, cls_names=None):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl*2//3, lineType=cv2.LINE_AA)
    if label:
        if len(label.split(' ')) > 1:
            label = label.split(' ')[-1]
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 6, [225, 255, 255], thickness=tf//2, lineType=cv2.LINE_AA)

    if cls_names is not None:  #This block is enabled from demo.py. Have to be enabled for
        score = x[4] * x[-2]
        text = '{} : {:.1f}%'.format(cls_names[int(x[-1])], score * 100)
        txt_color = color
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        txt_bk_color = (np.array(color) * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            im,
            (c1[0], c1[1] + 1),
            (c1[0] + txt_size[0] + 1, c1[1] + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(im, text, (c1[0], c1[1] + txt_size[1]), font, 0.4, txt_color, thickness=1)

    if human_pose:
        plot_skeleton_kpts(im, kpts, steps, orig_shape=orig_shape)
    elif object_pose:
        plot_object_pose(im, im_cuboid, im_mask, pose, cad_models, camera_matrix, color, label, block_x, block_y, orig_shape=orig_shape)


def plot_object_pose(im, im_cuboid, im_mask, pose, cad_models, camera_matrix, color, label, block_x, block_y, orig_shape=None):

    img_2dod = copy.deepcopy(im)
    rotation = pose['rotation_vec']
    translation = pose['translation_vec']
    xy = pose['xy']

    img_cuboid = cv2.circle(im_cuboid, (int(xy[0])+block_x, int(xy[1])+block_y), 3, (0, 0, 255), -1)
    cad_model_2d = project_3d_2d(pts_3d=cad_models.class_to_model[int(label)],
                                 rotation_vec=rotation, translation_vec=translation, camera_matrix=camera_matrix)
    cad_model_2d = cad_model_2d.astype(np.int32)
    cad_model_2d[:, 0][cad_model_2d[:,0] >= orig_shape[1]] = orig_shape[1] - 1
    cad_model_2d[:, 1][cad_model_2d[:,1] >= orig_shape[0]] = orig_shape[0] - 1
    cad_model_2d[cad_model_2d < 0] = 0
    cad_model_2d[:, 0] += block_x
    cad_model_2d[:, 1] += block_y

    im_mask[cad_model_2d[:, 1], cad_model_2d[:, 0]] = color
    img_mask = cv2.circle(im_mask, (int(xy[0])+block_x, int(xy[1])+block_y), 3, (0, 0, 255), -1)

    cuboid_corners_2d = project_3d_2d(pts_3d=cad_models.models_corners[int(label)],
                                rotation_vec=rotation, translation_vec=translation, camera_matrix=camera_matrix
    )
    cuboid_corners_2d[:, 0] += block_x
    cuboid_corners_2d[:, 1] += block_y
    img_cuboid = draw_cuboid_2d(img=img_cuboid, cuboid_corners=cuboid_corners_2d, colour=color)

    #img_2dod = draw_bbox_2d(img_2dod, bbox, label, score, conf=0.6, thickness=2)

    return img_mask, img_cuboid, img_2dod



def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]

    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % orig_shape[1] == 0 or y_coord % orig_shape[0] == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%orig_shape[1] == 0 or pos1[1]%orig_shape[0]==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % orig_shape[1] == 0 or pos2[1] % orig_shape[0] == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

    return im

def plot_one_box_PIL(box, im, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image 'im' using PIL
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    line_thickness = line_thickness or max(int(min(im.size) / 200), 2)
    draw.rectangle(box, width=line_thickness, outline=tuple(color))  # plot
    if label:
        fontsize = max(round(max(im.size) / 40), 12)
        font = ImageFont.truetype("Arial.ttf", fontsize)
        txt_width, txt_height = font.getsize(label)
        draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=tuple(color))
        draw.text((box[0], box[1] - txt_height + 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(im)



def output_to_target(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        kpts = o[:,6:]
        o = o[:,:6]
        for index, (*box, conf, cls) in enumerate(o.cpu().numpy()):
            targets.append([i, cls, *list(*xyxy2cxcywh(np.array(box)[None])), conf, *list(kpts.cpu().numpy()[index])])
    return np.array(targets)


def plot_images(images, targets, paths=None, fname='images.png', names=None, max_size=640, max_subplots=16, human_pose=False, object_pose=False,
                steps=2, orig_shape=None, dataset=None, data_index=None):
    # Plot image grid with labels
    if hasattr(dataset, 'cad_models'):
        cad_models = dataset.cad_models
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()  #create a copy
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()  # create a copy

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)
    #print(h,w)
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    if object_pose:
        mosaic_cuboid = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
        mosaic_mask = copy.deepcopy(mosaic)
    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        if object_pose:
            if isinstance(dataset, YCBVDataset):
                img_index = list(dataset.imgs_coco)[data_index[i]]
                image_folder = dataset.imgs_coco[int(img_index)]['image_folder']
                if int(image_folder)<60:
                    camera_matrix = cad_models.camera_matrix['camera_uw']
                else:
                    camera_matrix = cad_models.camera_matrix['camera_cmu']
            else :
                camera_matrix = cad_models.camera_matrix

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if object_pose:
            mosaic_cuboid[block_y:block_y + h, block_x:block_x + w, :] = img
            mosaic_mask[block_y:block_y + h, block_x:block_x + w, :] = img

        if len(targets) > 0:
            image_targets = targets[i]
            # valid_targets = np.sum(np.any(image_targets!=0, axis=1))
            # image_targets = image_targets[:valid_targets, :]
            boxes = cxcywh2xyxy(image_targets[:, 1:5]).T
            classes = image_targets[:, 0].astype('int')
            if human_pose:
                labels = image_targets.shape[1] == 39
            elif object_pose:
                labels = image_targets.shape[1] == 14
            else:
                labels = image_targets.shape[1] == 5   # labels if no conf column
            conf = None if labels else image_targets[:, 6]  # check for confidence presence (label vs pred)
            if human_pose:
                if conf is None:
                    kpts = image_targets[:, 5:].T   #kpts for GT
                else:
                    kpts = image_targets[:, 6:].T    #kpts for prediction
            elif object_pose:
                image_targets[:, -3:-1] *= scale_factor
            else:
                kpts = None

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale_factor < 1:  # absolute coords need scale if image scales
                    boxes *= scale_factor
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] += block_y

            if human_pose and kpts.shape[1]:
                if kpts.max()<1.01:
                    kpts[list(range(0,len(kpts),steps))] *=w # scale to pixels
                    kpts[list(range(1,len(kpts),steps))] *= h
                elif scale_factor < 1:  # absolute coords need scale if image scales
                    kpts *= scale_factor
                kpts[list(range(0, len(kpts), steps))] += block_x
                kpts[list(range(1, len(kpts), steps))] += block_y

            for j, box in enumerate(boxes.T):
                if not (np.all(box[0::2]%w == 0) and np.all(box[1::2]%h ==0)) :
                    cls = int(classes[j])
                    color = colors(cls)
                    cls = names[cls] if names else cls
                    if labels or conf[j] > 0.25:  # 0.25 conf thresh
                        label = '%s' % cls if labels else '%s %.1f' % (cls, conf[j])
                        if human_pose:
                            plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl, human_pose=human_pose, kpts=kpts[:,j], steps=steps, orig_shape=(h,w))
                        elif object_pose:
                            pose = {}
                            pose['xy'] = copy.deepcopy(image_targets[j][-3:-1])
                            rotation_vec, translation_vec = decode_rotation_translation(image_targets[j], camera_matrix=camera_matrix)
                            pose["rotation_vec"] = rotation_vec
                            pose["translation_vec"] = translation_vec
                            plot_one_box(box, mosaic, im_cuboid=mosaic_cuboid, im_mask=mosaic_mask, label=label, color=color, line_thickness=tl, object_pose=object_pose,
                                         orig_shape=(h, w), cad_models=cad_models, camera_matrix=camera_matrix, pose=pose, block_x=block_x, block_y=block_y)
                        else:
                            plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl, human_pose=human_pose, orig_shape=orig_shape)
                        #cv2.imwrite(Path(paths[i]).name.split('.')[0] + "_box_{}.".format(j) + Path(paths[i]).name.split('.')[1], mosaic[:,:,::-1]) # used for debugging the dataloader pipeline

        # Draw image filename labels
        if paths:
            label = Path(paths[i]).name[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]

            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 6, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname:
        r = min(1280. / max(h, w) / ns, 1.0)  # ratio to limit image size
        mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))  # cv2 save
        mosaic = mosaic[:,:,::-1]
        Image.fromarray(mosaic).save(fname)  # PIL save
        if object_pose:
            mosaic_mask = mosaic_mask[:, :, ::-1]
            Image.fromarray(mosaic_mask).save(fname.replace(".png", "_mask.png"))  # PIL save
            mosaic_cuboid = mosaic_cuboid[:, :, ::-1]
            Image.fromarray(mosaic_cuboid).save(fname.replace(".png", "_cuboid.png"))  # PIL save
    return mosaic
