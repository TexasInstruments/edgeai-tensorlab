import cv2
import numpy as np
from torch import is_tensor
import copy
import os

camera_matrix = np.array([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0], dtype=np.float32)

colours = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 125, 255), 
(0, 255, 125), (125, 0, 255), (255, 0, 125), (125, 255, 0), (255, 125, 0), (125, 255, 255), (255, 125, 255),
(255, 255, 125), (255, 255, 255)]

r_h = 640 / 640
r_w = 640 / 640

px = 325.26110
py = 242.04899

fx = 572.41140
fy = 573.57043


def draw_bbox_2d(img, box, label, score, conf = 0.6, colours=colours, thickness=1, gt=False):

    cls_id = int(label)
    if score < conf:
        return
    x0, y0 = int(box[0]), int(box[1])
    x1, y1 = int(box[0] + box[2]), int(box[1] + box[3])

    color = colours[cls_id]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)

    return img


def draw_cuboid_2d(img, cuboid_corners, colour = (0, 255, 0), thickness = 2):
    box = np.copy(cuboid_corners).astype(np.int32)

    #front??? to check
    cv2.line(img, box[0], box[1], colour, thickness)
    cv2.line(img, box[1], box[2], colour, thickness)
    cv2.line(img, box[2], box[3], colour, thickness)
    cv2.line(img, box[0], box[3], colour, thickness)
    #back
    cv2.line(img, box[4], box[5], colour, thickness)
    cv2.line(img, box[5], box[6], colour, thickness)
    cv2.line(img, box[6], box[7], colour, thickness)
    cv2.line(img, box[4], box[7], colour, thickness)
    #sides
    cv2.line(img, box[0], box[4], colour, thickness)
    cv2.line(img, box[1], box[5], colour, thickness)
    cv2.line(img, box[2], box[6], colour, thickness)
    cv2.line(img, box[3], box[7], colour, thickness)

    return img


def project_cuboid(cuboid_corners, rotation_vec, translation_vec, camera_matrix):
   cuboid_corners_2d, _ = cv2.projectPoints(
       objectPoints=cuboid_corners,
       rvec=rotation_vec,
       tvec=translation_vec,
       cameraMatrix=camera_matrix.reshape((3,3)),
       distCoeffs=None
    ) 
   cuboid_corners_2d = np.squeeze(cuboid_corners_2d)

   return cuboid_corners_2d


def project_cad_model(cad_model, rotation_vec, translation_vec, camera_matrix):
   cad_model_2d, _ = cv2.projectPoints(
       objectPoints=cad_model,
       rvec=rotation_vec,
       tvec=translation_vec,
       cameraMatrix=camera_matrix.reshape((3,3)),
       distCoeffs=None
    )
   cad_model_2d = np.squeeze(cad_model_2d)

   return cad_model_2d


def draw_6d_pose(img, data_list, class_to_cuboid=None, camera_matrix=camera_matrix, colours=colours, conf = 0.6, class_to_model=None, gt=True, out_dir=None, id=None):

    if is_tensor(img):
        img_cuboid = copy.deepcopy(img).cpu().numpy().transpose(1, 2, 0)
        img_cuboid = np.asarray(img_cuboid, dtype=np.uint8)
        img_cuboid = np.ascontiguousarray(img_cuboid)
        img_mask = copy.deepcopy(img_cuboid)
        img_2dod = copy.deepcopy(img_cuboid)

    # if is_tensor(poses):
    #     poses = poses.cpu().numpy()
   
    for pose in data_list:
        #Rotation matrix is recovered using the formula given in the article
        #https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f
        pose_type = "gt" if gt else "pred"
        if pose['missing_det']:
            continue
        score = pose['score'] if pose_type == "pred" else 1.0
        if score < conf:
            continue
        rotation, translation, bbox, xy,  label = \
            np.array(pose['rotation_{}'.format(pose_type)]), np.array(pose['translation_{}'.format(pose_type)]), pose['bbox_{}'.format(pose_type)], pose['xy_{}'.format(pose_type)], pose['category_id']
        colour = colours[label]

        img_cuboid = cv2.circle(img_cuboid, (int(xy[0]), int(xy[1])), 3, (0, 0, 255), -1)

        cad_model_2d = project_cad_model(class_to_model[label], rotation, translation, camera_matrix)
        cad_model_2d = cad_model_2d.astype(np.int32)
        cad_model_2d[cad_model_2d >= 640] = 639
        cad_model_2d[cad_model_2d < 0] = 0
        img_mask[cad_model_2d[:, 1], cad_model_2d[:, 0]] = colour
        img_mask = cv2.circle(img_mask, (int(xy[0]), int(xy[1])), 3, (0, 0, 255), -1)

        cuboid_corners_2d = project_cuboid(cuboid_corners=class_to_cuboid[label],
            rotation_vec=rotation, translation_vec=translation, camera_matrix=camera_matrix
        )
        img_cuboid = draw_cuboid_2d(img=img_cuboid, cuboid_corners=cuboid_corners_2d, colour=colour)

        img_2dod = draw_bbox_2d(img_2dod, bbox, label, score, conf=0.6, thickness=1, gt=gt)

    outfile_pose = os.path.join(out_dir, "vis_pose", "{:012}_{}_pose.png".format(id, pose_type))
    outfile_mask = os.path.join(out_dir, "vis_pose", "{:012}_{}_mask.png".format(id, pose_type))
    outfile_2d_od = os.path.join(out_dir, "vis_pose", "{:012}_{}_2d_od.png".format(id, pose_type))
    cv2.imwrite(outfile_pose, img_cuboid)
    cv2.imwrite(outfile_mask, img_mask)
    cv2.imwrite(outfile_2d_od, img_2dod)

