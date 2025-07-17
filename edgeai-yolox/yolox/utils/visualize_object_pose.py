import cv2
import numpy as np
from torch import is_tensor
import copy
import os
import matplotlib

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        self.palette = [self.hex2rgb(c) for c in matplotlib.colors.TABLEAU_COLORS.values()]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def draw_bbox_2d(img, box, label, score, conf = 0.6, thickness=1, gt=False):

    cls_id = int(label)
    if score < conf:
        return
    x0, y0 = int(box[0]), int(box[1])
    x1, y1 = int(box[0] + box[2]), int(box[1] + box[3])

    if gt:
        color = (0, 255, 0)
    else:
        color = colors(cls_id)

    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    if gt:
        cv2.putText(img, str(label), ((x0+x1)//2, (y0+y1)//2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=thickness)
    else:
        cv2.putText(img, str(label), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=thickness)
    return img


def draw_cuboid_2d(img, cuboid_corners, colour = (0, 255, 0), thickness = 2):
    box = np.copy(cuboid_corners).astype(np.int32)
    box = [tuple(kpt) for kpt in box]
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


def project_3d_2d(pts_3d, rotation_vec, translation_vec, camera_matrix):
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    xformed_3d = np.matmul(pts_3d, rotation_mat.T) + translation_vec
    xformed_3d[:,:3] = xformed_3d[:,:3]/xformed_3d[:,2:3]
    projected_2d = np.matmul(xformed_3d, camera_matrix.reshape((3, 3)).T)[:, :2]
    #projected_2d_corrected = copy.deepcopy(projected_2d)
    #projected_2d_corrected = cv2.undistortPoints(projected_2d_corrected, camera_matrix.reshape(3,3), np.array([0.04112172, -0.4798174, 0.0, 0.0, 1.890084 ]))
    #projected_2d_corrected = np.squeeze(projected_2d_corrected)
    #projected_2d_corrected[:, 0] = camera_matrix[0] * projected_2d_corrected[:, 0] + camera_matrix[2]
    #projected_2d_corrected[:, 1] = camera_matrix[4] * projected_2d_corrected[:, 1] + camera_matrix[5]

    return projected_2d


def draw_6d_pose(img, data_list , camera_matrix, class_to_cuboid=None, conf = 0.6, class_to_model=None, gt=True, out_dir=None, id=None, img_cuboid=None, img_mask=None, img_2dod=None):

    if is_tensor(img):
        img_temp = copy.deepcopy(img).cpu().numpy().transpose(1, 2, 0)
        img_temp = np.asarray(img_temp, dtype=np.uint8)
        img_temp = np.ascontiguousarray(img_temp)
        if img_mask is None:
            img_mask = copy.deepcopy(img_temp)
        if img_cuboid is None:
            img_cuboid = copy.deepcopy(img_temp)
        if img_2dod is None:
            img_2dod = copy.deepcopy(img_temp)

    for pose in data_list:
        #Rotation matrix is recovered using the formula given in the article
        #https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f
        pose_type = "gt" if gt else "pred"
        if pose['missing_det'] and not gt:
            continue
        score = pose['score'] if pose_type == "pred" else 1.0
        if score < conf:
            continue
        rotation, translation, bbox, xy,  label = \
            np.array(pose['rotation_{}'.format(pose_type)]), np.array(pose['translation_{}'.format(pose_type)]), \
                        pose['bbox_{}'.format(pose_type)], pose['xy_{}'.format(pose_type)], pose['category_id']
        if gt:
            colour = (0, 255, 0)
        else:
            colour = colors(label)

        img_cuboid = cv2.circle(img_cuboid, (int(xy[0]), int(xy[1])), 3, (0, 0, 255), -1)

        cad_model_2d = project_3d_2d(class_to_model[label], rotation, translation, camera_matrix)
        cad_model_2d = cad_model_2d.astype(np.int32)
        cad_model_2d[:, 0][cad_model_2d[:, 0] >= img.shape[2]] = img.shape[2] - 1
        cad_model_2d[:, 1][cad_model_2d[:, 1] >= img.shape[1]] = img.shape[1] - 1
        cad_model_2d[cad_model_2d < 0] = 0
        img_mask[cad_model_2d[:, 1], cad_model_2d[:, 0]] = colour
        img_mask = cv2.circle(img_mask, (int(xy[0]), int(xy[1])), 3, (0, 0, 255), -1)

        cuboid_corners_2d = project_3d_2d(class_to_cuboid[label], rotation, translation, camera_matrix)
        img_cuboid = draw_cuboid_2d(img=img_cuboid, cuboid_corners=cuboid_corners_2d, colour=colour)

        img_2dod = draw_bbox_2d(img_2dod, bbox, label, score, conf=0.6, thickness=2, gt=gt)

    outfile_pose = os.path.join(out_dir, "vis_pose", "{:012}_{}_pose.png".format(id, pose_type))
    outfile_mask = os.path.join(out_dir, "vis_pose", "{:012}_{}_mask.png".format(id, pose_type))
    outfile_2d_od = os.path.join(out_dir, "vis_pose", "{:012}_{}_2d_od.png".format(id, pose_type))
    if not gt:
        cv2.imwrite(outfile_pose, img_cuboid)
        cv2.imwrite(outfile_mask, img_mask)
        cv2.imwrite(outfile_2d_od, img_2dod)
    return img_cuboid, img_mask, img_2dod

