import cv2
import numpy as np
from torch import is_tensor

def draw_cuboid_2d(img, cuboid_corners, colour = (0, 255, 0), thickness = 2):
    box = np.copy(cuboid_corners).astype(np.int32)

    #upper level
    cv2.line(img, box[0], box[1], colour, thickness)
    cv2.line(img, box[1], box[2], colour, thickness)
    cv2.line(img, box[2], box[3], colour, thickness)
    cv2.line(img, box[0], box[3], colour, thickness)
    #lower level
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
       cameraMatrix=camera_matrix
    ) 
   cuboid_corners_2d = np.squeeze(cuboid_corners_2d)

   return cuboid_corners_2d

def draw_predictions(img, predictions, num_classes, class_to_cuboid, camera_matrix, colours):
    if is_tensor(predictions):
        predictions = predictions.numpy()

    for prediction in predictions:
        obj_class = np.argmax(prediction[-num_classes:])
        colour = colours(obj_class)
        rotation_vec = prediction[5:8]
        translation_vec = prediction[8:11]

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

def draw_ground_truths(img, ground_truths, class_to_cuboid, camera_matrix, colour=(0, 255, 0)):
    if is_tensor(ground_truths):
        ground_truths = ground_truths.numpy()

    for ground_truth in ground_truths:
        obj_class = ground_truth[0]
        rotation_vec = ground_truth[5:8]
        translation_vec = ground_truth[8:11]

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