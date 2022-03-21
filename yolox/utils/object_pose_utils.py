import torch
import numpy as np
from os import path
from plyfile import PlyData
from loguru import logger
from math import cos, sin
from .dist import get_local_rank
from .visualize_pose import class_to_cuboid

#Order is same as https://github.com/ybkscht/EfficientPose/blob/main/generators/occlusion.py since we use the same dataset
class_to_name = {0: "ape", 1: "can", 2: "cat", 3: "driller", 4: "duck", 5: "eggbox", 6: "glue", 7: "holepuncher", 8: "benchvise", 9: "bowl", 10: "cup", 11: "iron", 12: "lamp", 13: "phone", 14: "cam"}

def calculate_model_rotation(point_cloud, rvec):
    #rvec = rvec.cpu()
    point_cloud = point_cloud.to(device="cuda:{}".format(get_local_rank()))
    theta = float(rvec.norm(dim = 0))
    if theta != 0:
        k = rvec / theta
    else:
        k = 0 * rvec
    rows = int(point_cloud.shape[0])
    k_cross = torch.tensor([]).to(device="cuda:{}".format(get_local_rank()))
    for _ in range(rows):
        k_cross = torch.cat((k_cross, k))
    k_cross = k_cross.reshape(rows, 3)
    k = k.reshape(1, 3)

    points_transformed = point_cloud * cos(theta) + torch.cross(k_cross, point_cloud) * sin(theta) + k_cross * torch.mm(point_cloud, k.transpose(0, 1)) * (1 - cos(theta))
    return points_transformed

def load_models(models_datapath, class_to_name=class_to_name):
    class_to_model = {class_id: None for class_id in class_to_name.keys()}
    logger.info("Loading 3D models...")

    for class_id, name in class_to_name.items():
        file = "obj_{:02}.ply".format(class_id + 1)
        model_datapath = path.join(models_datapath, file)

        if not path.isfile(model_datapath):
            logger.warning(
                "The file {} model for class {} was not found".format(file, name)
            )
            continue

        model_3D = load_model_point_cloud(model_datapath)
        class_to_model[class_id] = torch.tensor(model_3D, requires_grad=False).half()

    return class_to_model 

def load_model_point_cloud(datapath):
    model = PlyData.read(datapath)
                                  
    vertex = model['vertex']
    points = np.stack([vertex[:]['x'], vertex[:]['y'], vertex[:]['z']], axis = -1).astype(np.float64)
        
    return points