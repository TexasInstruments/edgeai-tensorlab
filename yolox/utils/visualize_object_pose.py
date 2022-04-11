import cv2
import numpy as np
from torch import is_tensor
import copy

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


def draw_bbox_2d(img, boxes,  conf = 0.6, colours=colours, thickness=1, gt=False):
    if is_tensor(img):
        img = copy.deepcopy(img).cpu().numpy().transpose(1, 2, 0)
        img = np.asarray(img, dtype=np.uint8)
        img = np.ascontiguousarray(img)

    for i in range(len(boxes)):
        box = boxes[i][0:4]
        cls_id = int(boxes[i][4])
        if not gt:
            score = boxes[i][5]*boxes[i][6]
            cls_id = int(boxes[i][-1])
            if score < conf:
                continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = colours[cls_id]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

    return img


def draw_cuboid_2d(img, cuboid_corners, colour = (0, 255, 0), thickness = 2):
    box = np.copy(cuboid_corners).astype(np.int32)

    #front
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


def draw_predictions(img, predictions, class_to_cuboid=None, camera_matrix=camera_matrix, colours=colours, conf = 0.6, class_to_model=None):

    if is_tensor(img):
        img = copy.deepcopy(img).cpu().numpy().transpose(1, 2, 0)
        img = np.asarray(img, dtype=np.uint8)
        img = np.ascontiguousarray(img)
        img_mask = copy.deepcopy(img)

    if is_tensor(predictions):
        predictions = predictions.cpu()
        predictions = predictions.numpy()
   
    for prediction in predictions:
        if prediction[4]*prediction[-2] < conf:
            continue
        obj_class = int(prediction[-1])
        colour = colours[obj_class]
        
        #Rotation matrix is recovered using the formula given in the article
        #https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f
        r1 = np.expand_dims(prediction[5:8].astype(np.float64), axis=1)
        r2 = np.expand_dims(prediction[8:11].astype(np.float64), axis=1)
        r3 = np.cross(r1.T, r2.T).T
        translation_vec = prediction[11:14].astype(np.float64)
        #Tz was previously scaled down by 100 (converted from cm to m)
        #Tx and Ty are recovered using the formula given on page 5 of the the paper: https://arxiv.org/pdf/2011.04307.pdf
        #px, py, fx and fy are currently hard-coded for LINEMOD dataset
        tz = prediction[13].astype(np.float64) * 100.0
        #print("prediction",obj_class, tz)
        tx = ((prediction[11].astype(np.float64) / r_w ) - px) * tz / fx
        ty = ((prediction[12].astype(np.float64) / r_h ) - py) * tz / fy
        img = cv2.circle(img, (int(prediction[11]), int(prediction[12])), 3, (0, 0, 255), -1)
        rotation_mat = np.hstack((r1, r2, r3))
        rotation_vec, _ = cv2.Rodrigues(rotation_mat)
        translation_vec[0] = tx
        translation_vec[1] = ty
        translation_vec[2] = tz

        cad_model_2d = project_cad_model(class_to_model[obj_class], rotation_vec, translation_vec, camera_matrix)
        cad_model_2d = cad_model_2d.astype(np.int32)
        cad_model_2d[cad_model_2d >= 640] = 639
        cad_model_2d[cad_model_2d < 0] = 0
        img_mask[cad_model_2d[:, 1], cad_model_2d[:, 0]] = colour
        #print(int(prediction[11]), int(prediction[12]))
        img_mask = cv2.circle(img_mask, (int(prediction[11]), int(prediction[12])), 3, (0, 0, 255), -1)

        cuboid_corners_2d = project_cuboid(
            cuboid_corners=class_to_cuboid[obj_class],
            rotation_vec=rotation_vec,
            translation_vec=translation_vec,
            camera_matrix=camera_matrix
        )

        draw_cuboid_2d(
            img=img,
            cuboid_corners=cuboid_corners_2d,
            colour=colour
        )
    return img, img_mask

def draw_ground_truths(img, ground_truths, class_to_cuboid=None, camera_matrix=camera_matrix, colours=colours, class_to_model=None):

    if is_tensor(img):
        img = copy.deepcopy(img).cpu().numpy().transpose(1, 2, 0)
        img = np.asarray(img, dtype=np.uint8)
        img = np.ascontiguousarray(img)
        img_mask = copy.deepcopy(img)

    if is_tensor(ground_truths):
        ground_truths = ground_truths.numpy()

    for ground_truth in ground_truths:
        for obj in ground_truth:
            obj_class = int(obj[4])
            r1 = np.expand_dims(obj[5:8], axis=1)
            r2 = np.expand_dims(obj[8:11], axis=1)
            r3 = np.cross(r1.T, r2.T).T
            translation_vec = copy.deepcopy(obj[11:14])
            tz = obj[13] * 100.0
            tx = ((obj[11] / r_w) - px )* tz / fx
            ty = ((obj[12] / r_h) - py )* tz / fy
            img = cv2.circle(img, (int(obj[11]), int(obj[12])), 3, (0, 0, 255), -1)

            rotation_mat = np.hstack((r1, r2, r3))
            rotation_vec, _ = cv2.Rodrigues(rotation_mat)
            translation_vec[0] = tx
            translation_vec[1] = ty
            translation_vec[2] = tz

            #rotation_vec = obj[5:8]
            #translation_vec = obj[8:11]
            colour = colours[obj_class]
            cad_model_2d = project_cad_model(class_to_model[obj_class], rotation_vec, translation_vec, camera_matrix)
            cad_model_2d = cad_model_2d.astype(np.int32)
            cad_model_2d[cad_model_2d >= 640] = 639
            cad_model_2d[cad_model_2d < 0] = 0
            img_mask[cad_model_2d[:, 1], cad_model_2d[:, 0]] = colour
            img_mask = cv2.circle(img_mask, (int(obj[11]), int(obj[12])), 3, (0, 0, 255), -1)
            cuboid_corners_2d = project_cuboid(
                cuboid_corners=class_to_cuboid[int(obj_class)],
                rotation_vec=rotation_vec,
                translation_vec=translation_vec,
                camera_matrix=camera_matrix
            )

            draw_cuboid_2d(
                img=img,
                cuboid_corners=cuboid_corners_2d,
                colour=colour
            )

    return img, img_mask