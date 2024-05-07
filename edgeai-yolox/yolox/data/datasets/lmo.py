#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from loguru import logger

import cv2
import numpy as np
from math import acos, sin, sqrt, copysign
from pycocotools.coco import COCO
from plyfile import PlyData
import yaml, json

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset

class CADModelsLM():
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "lmo")
        self.data_dir = data_dir
        self.cad_models_path = os.path.join(self.data_dir, "models")
        self.class_to_name_orig = {
                              0: "ape",  4: "can", 5: "cat", 7: "driller",
                              8: "duck", 9: "eggbox", 10: "glue", 11: "holepuncher",
                             }
        self.class_to_name = {                               #class map used for training
                             0: "ape",  1: "can", 2: "cat", 3: "driller",
                             4: "duck", 5: "eggbox", 6: "glue", 7: "holepuncher",
                             }
        self.class_map = {                            #class map used for training
                             0: 0, 4: 1, 5: 2,
                             7: 3, 8: 4, 9: 5,
                             10: 6, 11: 7,
                             }
        self.models_dict_path = os.path.join(self.cad_models_path, "models_info.json")
        with open(self.models_dict_path) as foo:
            self.models_dict  = json.load(foo)
        self.class_to_model = self.load_cad_models()
        self.class_to_sparse_model = self.create_sparse_models()
        self.models_corners, self.models_diameter = self.get_models_params()
        self.symmetric_objects = {9: "eggbox", 10: "glue"}
        self.camera_matrix =  self.get_camera_params()

    def get_camera_params(self):
        camera_params_path = os.path.join(self.data_dir, "camera.json")
        with open(camera_params_path) as foo:
            camera_params = json.load(foo)
        camera_matrix = np.array([camera_params['fx'], 0, camera_params['cx'], 0.0, camera_params['fy'], camera_params['cy'], 0.0, 0.0, 1.0])
        return camera_matrix

    def load_cad_models(self):
        class_to_model = {class_id: None for class_id in self.class_map.values()}
        logger.info("Loading 3D models...")
        for class_id, name in self.class_to_name_orig.items():
            file = "obj_{:06}.ply".format(class_id + 1)
            cad_model_path = os.path.join(self.cad_models_path, file)

            if not os.path.isfile(cad_model_path):
                logger.warning(
                    "The file {} model for class {} was not found".format(file, name)
                )
                continue
            logger.info("Loading 3D model {}".format(name))
            class_to_model[self.class_map[class_id]] = self.load_model_point_cloud(cad_model_path)

        return class_to_model

    def load_model_point_cloud(self, datapath):
        model = PlyData.read(datapath)
        vertex = model['vertex']
        points = np.stack([vertex[:]['x'], vertex[:]['y'], vertex[:]['z']], axis=-1).astype(np.float64)
        return points

    def get_models_params(self):
        """
        Convert model corners from LINEMOD Occlusion format (min_x, min_y, min_z, size_x, size_y, size_z) to actual coordinates format of dimension (8,3)
        Return the corner coordinates and the diameters of each models
        """
        models_corners_3d = {}
        models_diameter = {}
        for model_id, model_param in self.models_dict.items():
            min_x, max_x = model_param['min_x'], model_param['min_x'] + model_param['size_x']
            min_y, max_y = model_param['min_y'], model_param['min_y'] + model_param['size_y']
            min_z, max_z = model_param['min_z'], model_param['min_z'] + model_param['size_z']
            corners_3d = np.array([
                [min_x, min_y, min_z],
                [min_x, min_y, max_z],
                [min_x, max_y, max_z],
                [min_x, max_y, min_z],
                [max_x, min_y, min_z],
                [max_x, min_y, max_z],
                [max_x, max_y, max_z],
                [max_x, max_y, min_z],
            ])
            cls_ind = self.class_map[int(model_id) - 1]
            models_corners_3d.update({cls_ind: corners_3d})
            models_diameter.update({cls_ind: model_param['diameter']})
        return models_corners_3d, models_diameter

    def create_sparse_models(self):
        class_to_sparse_model = {}
        for model_id in self.class_to_model.keys():
            sample_rate =len(self.class_to_model[model_id])//500
            #sparsely sample the model to have close to 500 points
            class_to_sparse_model.update({model_id : self.class_to_model[model_id][::sample_rate, :]})
        return class_to_sparse_model


class LMODataset(Dataset):
    """
    LINEMODOcclusion dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train.json",
        name="train",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        object_pose=False,
        symmetric_objects={5: "eggbox", 6: "glue"},
        base_dir = "lmo"
    ):
        """
        LINEMODOcclusion dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): LINEMOD Occlusion json file name
            name (str): LINEMOD Occlusion data name 
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), base_dir)
        self.data_dir = data_dir
        self.json_file = json_file
        self.object_pose = object_pose 

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.imgs = None
        self.imgs_coco = self.coco.imgs
        self.name = name
        self.img_size = img_size
        self.cad_models = CADModelsLM()
        self.models_corners, self.models_diameter = self.cad_models.models_corners, self.cad_models.models_diameter
        self.class_to_name = self.cad_models.class_to_name
        self.class_map = self.cad_models.class_map
        self.class_to_model = self.cad_models.class_to_model
        if preproc is not None:
            self.preproc = preproc
        self.annotations = self._load_coco_annotations()
        self.symmetric_objects = symmetric_objects
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.data_dir + "/img_resized_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for COCO"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        if self.object_pose:
            res = np.zeros((num_objs, 14))
        else:
            res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            if self.object_pose:
                obj_centre_2d = np.matmul(self.cad_models.camera_matrix.reshape(3,3), np.array(obj["T"])/obj["T"][2])[:2]  #rotation vec not required for the center point
                #res[ix, 11:14] = obj["T"]
                obj_centre_2d = np.squeeze(obj_centre_2d)
                res[ix, -3:-1] = obj_centre_2d
                res[ix, -1] = obj["T"][2] / 100.0
                #obj["R_aa"], _ = cv2.Rodrigues(np.array(obj["R"]).reshape(3,3))
                #obj["R_aa"] = np.squeeze(obj["R_aa"])
                #Use Gram-Schmidt to make the rotation representation continuous and in 6D
                #https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f
                R_gs = np.array(obj["R"]).reshape(3,3)
                obj["R_gs"] = np.squeeze(R_gs[:, :2].transpose().reshape(6, 1))
                res[ix, -9:-3] = obj["R_gs"]
            #print(res[ix, 11:13])
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        res[:, -3:-1] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:04}".format(id_) + ".png"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]
        img_index = list(self.imgs_coco)[index]
        image_folder = self.imgs_coco[int(index)]['image_folder']
        type = self.imgs_coco[img_index]['type']
        img_file = os.path.join(self.data_dir, self.name + '_' + type, image_folder, 'rgb', file_name)
        img = cv2.imread(img_file)
        assert img is not None

        return img

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, resized_info, _ = self.annotations[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        return img, res.copy(), img_info, index

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
