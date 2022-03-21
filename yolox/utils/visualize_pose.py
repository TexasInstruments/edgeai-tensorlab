import cv2
import numpy as np
from torch import is_tensor

#Based on code from https://github.com/ybkscht/EfficientPose/blob/main/utils/visualization.py

class_to_cuboid = [
    np.array([[-37.9343, -38.7996, 45.8845], [-37.9343, 38.7996, 45.8845], [37.9343, 38.7996, 45.8845], [37.9343, -38.7996, 45.8845],
    [-37.9343, -38.7996, -45.8845], [-37.9343, 38.7996, -45.8845], [37.9343, 38.7996, -45.8845], [37.9343, -38.7996, -45.8845]]),
    np.array([[-107.835, -60.9279, 109.705], [-107.835, 60.9279, 109.705], [107.835, 60.9279, 109.705], [107.835, -60.9279, 109.705],
    [-107.835, -60.9279, -109.705], [-107.835, 60.9279, -109.705], [107.835, 60.9279, -109.705], [107.835, -60.9279, -109.705]]),
    np.array([[-83.2162, -82.6591, 37.2364], [-83.2162, 82.6591, 37.2364], [83.2162, 82.6591, 37.2364], [83.2162, -82.6591, 37.2364],
    [-83.2162, -82.6591, -37.2364], [-83.2162, 82.6591, -37.2364], [83.2162, 82.6591, -37.2364], [83.2162, -82.6591, -37.2364]]),
    np.array([[-68.3297, -71.5151, 50.2485], [-68.3297, 71.5151, 50.2485], [68.3297, 71.5151, 50.2485], [68.3297, -71.5151, 50.2485],
    [-68.3297, -71.5151, -50.2485], [-68.3297, 71.5151, -50.2485], [68.3297, 71.5151, -50.2485], [68.3297, -71.5151, -50.2485]]),
    np.array([[-50.3958, -90.8979, 96.867], [-50.3958, 90.8979, 96.867], [50.3958, 90.8979, 96.867], [50.3958, -90.8979, 96.867],
    [-50.3958, -90.8979, -96.867], [-50.3958, 90.8979, -96.867], [50.3958, 90.8979, -96.867], [50.3958, -90.8979, -96.867]]),
    np.array([[-33.5054, -63.8165, 58.7283], [-33.5054, 63.8165, 58.7283], [33.5054, 63.8165, 58.7283], [33.5054, -63.8165, 58.7283],
    [-33.5054, -63.8165, -58.7283], [-33.5054, 63.8165, -58.7283], [33.5054, 63.8165, -58.7283], [33.5054, -63.8165, -58.7283]]),
    np.array([[-58.7899, -45.7556, 47.3112], [-58.7899, 45.7556, 47.3112], [58.7899, 45.7556, 47.3112], [58.7899, -45.7556, 47.3112],
    [-58.7899, -45.7556, -47.3112], [-58.7899, 45.7556, -47.3112], [58.7899, 45.7556, -47.3112], [58.7899, -45.7556, -47.3112]]),
    np.array([[-114.738, -37.7357, 104.001], [-114.738, 37.7357, 104.001], [114.738, 37.7357, 104.001], [114.738, -37.7357, 104.001],
    [-114.738, -37.7357, -104.001], [-114.738, 37.7357, -104.001], [114.738, 37.7357, -104.001], [114.738, -37.7357, -104.001]]),
    np.array([[-52.2146, -38.7038, 42.8485], [-52.2146, 38.7038, 42.8485], [52.2146, 38.7038, 42.8485], [52.2146, -38.7038, 42.8485],
    [-52.2146, -38.7038, -42.8485], [-52.2146, 38.7038, -42.8485], [52.2146, 38.7038, -42.8485], [52.2146, -38.7038, -42.8485]]),
    np.array([[-75.0923, -53.5375, 34.6207], [-75.0923, 53.5375, 34.6207], [75.0923, 53.5375, 34.6207], [75.0923, -53.5375, 34.6207],
    [-75.0923, -53.5375, -34.6207], [-75.0923, 53.5375, -34.6207], [75.0923, 53.5375, -34.6207], [75.0923, -53.5375, -34.6207]]),
    np.array([[-18.3605, -38.933, 86.4079], [-18.3605, 38.933, 86.4079], [18.3605, 38.933, 86.4079], [18.3605, -38.933, 86.4079],
    [-18.3605, -38.933, -86.4079], [-18.3605, 38.933, -86.4079], [18.3605, 38.933, -86.4079], [18.3605, -38.933, -86.4079]]),
    np.array([[-50.4439, -54.2485, 45.4], [-50.4439, 54.2485, 45.4], [50.4439, 54.2485, 45.4], [50.4439, -54.2485, 45.4],
    [-50.4439, -54.2485, -45.4], [-50.4439, 54.2485, -45.4], [50.4439, 54.2485, -45.4], [50.4439, -54.2485, -45.4]]),
    np.array([[-129.113, -59.241, 70.5662], [-129.113, 59.241, 70.5662], [129.113, 59.241, 70.5662], [129.113, -59.241, 70.5662],
    [-129.113, -59.241, -70.5662], [-129.113, 59.241, -70.5662], [129.113, 59.241, -70.5662], [129.113, -59.241, -70.5662]]),
    np.array([[-101.573, -58.8763, 106.558], [-101.573, 58.8763, 106.558], [101.573, 58.8763, 106.558], [101.573, -58.8763, 106.558],
    [-101.573, -58.8763, -106.558], [-101.573, 58.8763, -106.558], [101.573, 58.8763, -106.558], [101.573, -58.8763, -106.558]]),
    np.array([[-46.9591, -73.7167, 92.3737], [-46.9591, 73.7167, 92.3737], [46.9591, 73.7167, 92.3737], [46.9591, -73.7167, 92.3737],
    [-46.9591, -73.7167, -92.3737], [-46.9591, 73.7167, -92.3737], [46.9591, 73.7167, -92.3737], [46.9591, -73.7167, -92.3737]])
]
camera_matrix = np.array([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0])
colours = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 125, 255), 
(0, 255, 125), (125, 0, 255), (255, 0, 125), (125, 255, 0), (255, 125, 0), (125, 255, 255), (255, 125, 255),
(255, 255, 125), (255, 255, 255)]

r_h = 640 / 480
r_w = 640 / 640

px = 325.26110
py = 242.04899

fx = 572.41140
fy = 573.57043

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
       cameraMatrix=camera_matrix.reshape((3,3)),
       distCoeffs=None
    ) 
   cuboid_corners_2d = np.squeeze(cuboid_corners_2d)

   return cuboid_corners_2d

def draw_predictions(img, predictions, num_classes, class_to_cuboid=class_to_cuboid, camera_matrix=camera_matrix, colours=colours, conf = 0.5):
 
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
        r1 = prediction[5:8].astype(np.float64)
        r2 = prediction[8:11].astype(np.float64)
        r3 = np.cross(r1, r2)
        translation_vec = np.zeros_like(prediction[11:14].astype(np.float64))
        #Tz was previously scaled down by 100 (converted from cm to m)
        #Tx and Ty are recovered using the formula given on page 5 of the the paper: https://arxiv.org/pdf/2011.04307.pdf
        #px, py, fx and fy are currently hard-coded for LINEMOD dataset
        tz = prediction[13].astype(np.float64) * 100.0
        tx = ((prediction[11].astype(np.float64) / r_w) - px) * tz / fx
        ty = ((prediction[12].astype(np.float64) / r_h) - py) * tz / fy

        rotation_vec = np.vstack((r1, r2, r3))
        rotation_vec, _ = cv2.Rodrigues(rotation_vec)
        translation_vec[0] = tx
        translation_vec[1] = ty
        translation_vec[2] = tz

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

def draw_ground_truths(img, ground_truths, class_to_cuboid=class_to_cuboid, camera_matrix=camera_matrix, colour=(0, 255, 0)):
    if is_tensor(ground_truths):
        ground_truths = ground_truths.numpy()

    for ground_truth in ground_truths:
        for obj in ground_truth:
            obj_class = obj[4] - 1
            r1 = obj[5:8]
            r2 = obj[8:11]
            r3 = np.cross(r1, r2)
            translation_vec = np.zeros_like(obj[11:14])
            tz = obj[13] * 100.0
            tx = ((obj[11] / r_w) - px) * tz / fx
            ty = ((obj[12] / r_h) - py) * tz / fy

            rotation_vec = np.vstack((r1, r2, r3))
            rotation_vec, _ = cv2.Rodrigues(rotation_vec)
            translation_vec[0] = tx
            translation_vec[1] = ty
            translation_vec[2] = tz
            
            rotation_vec = obj[5:8]
            translation_vec = obj[8:11]
            colour = colour

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