# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numbers
import os
import random
import json_tricks as json
import shutil
import tempfile
import numpy as np
from collections import OrderedDict, defaultdict
from colorama import Fore
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from plyfile import PlyData
from sklearn.neighbors import KDTree

from ..utils import *
# from ..utils import *
from ..datasets.dataset_base import *
# from .dataset_base import *

__all__ = ['YCBV', '_get_mapping_id_name']

def _get_mapping_id_name(imgs):
    """
    Args:
        imgs (dict): dict of image info.

    Returns:
        tuple: Image name & id mapping dicts.

        - id2name (dict): Mapping image id to name.
        - name2id (dict): Mapping image name to id.
    """
    id2name = {}
    name2id = {}
    for image_id, image in imgs.items():
        file_name = image['file_name']
        id2name[image_id] = file_name
        name2id[file_name] = image_id

    return id2name, name2id


class YCBV(DatasetBase):
    def __init__(self, download=False, num_frames=None, name="ycbv", **kwargs):
        super().__init__(num_frames=num_frames, name=name, **kwargs)
        self.force_download = True if download == 'always' else False
        assert 'path' in self.kwargs and 'split' in self.kwargs, 'kwargs must have path and split'
        path = self.kwargs['path']
        split = self.kwargs['split']
        if download:
            self.download(path, split)
        #
        self.class_to_name = {
               0: "002_master_chef_can" , 1: "003_cracker_box" ,  2: "004_sugar_box" , 3: "005_tomato_soup_can",  4: "006_mustard_bottle",
                5: "007_tuna_fish_can",  6: "008_pudding_box" , 7: "009_gelatin_box", 8: "010_potted_meat_can",  9: "011_banana",
                10: "019_pitcher_base", 11: "021_bleach_cleanser",  12: "024_bowl", 13: "025_mug", 14: "035_power_drill",
                15: "036_wood_block", 16: "037_scissors", 17: "040_large_marker", 18: "051_large_clamp", 19: "052_extra_large_clamp",
                20: "061_foam_brick"
                }
        dataset_folders = os.listdir(self.kwargs['path'])
        if 'annotations' not in dataset_folders:
            self.ycb2coco(path, split)  #convert ycb annotations to coco format
        annotations_dir = os.path.join(self.kwargs['path'], 'annotations')

        shuffle = self.kwargs.get('shuffle', False)
        image_split_dirs = os.listdir(self.kwargs['path'])
        assert self.kwargs['split'] in image_split_dirs, f'invalid path to coco dataset images/split {kwargs["split"]}'
        self.image_dir = os.path.join(self.kwargs['path'], self.kwargs['split'])
        self.annotation_file = os.path.join(annotations_dir, 'instances_{}.json'.format(split))
        self.coco_dataset = COCO(self.annotation_file)
        max_frames = len(self.coco_dataset.imgs)
        num_frames = self.kwargs.get('num_frames', None)
        num_frames = min(num_frames, max_frames) if num_frames is not None else max_frames

        imgs_list = list(self.coco_dataset.imgs.items())
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(imgs_list)

        self.coco_dataset.imgs = {k:v for k,v in imgs_list[:num_frames]}

        self.cats = [
            cat['name'] for cat in self.coco_dataset.loadCats(self.coco_dataset.getCatIds())
        ]

        self.classes = ['__background__'] + self.cats
        self.kwargs['num_classes'] = self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(self.cats, self.coco_dataset.getCatIds()))
        self.img_ids = self.coco_dataset.getImgIds()

        self.num_frames = self.kwargs['num_frames'] = num_frames
        self.tempfiles = []
        self.cad_models = CADModelsYCB(data_dir=self.kwargs['path'])
        self.symmetric_objects = self.cad_models.symmetric_objects
        self.models_dict = self.cad_models.models_dict
        self.models_corners, self.class_to_diameter = self.cad_models.models_corners, self.cad_models.models_diameter
        self.class_to_model = self.cad_models.class_to_model
        self.ann_info = {}
        self.id2name, self.name2id = _get_mapping_id_name(self.coco_dataset.imgs)
        # create dataset_info
        with open(self.annotation_file) as afp:
            self.dataset_store = json.load(afp)
        #
        self.kwargs['dataset_info'] = self.get_dataset_info()

    def get_dataset_info(self):
        if 'dataset_info' in self.kwargs:
            return self.kwargs['dataset_info']
        #
        # return only info and categories for now as the whole thing could be quite large.
        dataset_store = dict()
        for key in ('info', 'categories'):
            if key in self.dataset_store.keys():
                dataset_store.update({key: self.dataset_store[key]})
            if key == 'categories':
                for idx in range(len(dataset_store[key])):
                    dataset_store[key][idx] = dict(dataset_store[key][idx])
            elif key == 'info':
                dataset_store[key] = dict(dataset_store[key])
            #
        #
        dataset_store.update(dict(color_map=self.get_color_map()))        
        return dataset_store

    def download(self, path, split):
        root = path
        images_folder = os.path.join(path, split)
        annotations_folder = os.path.join(path, 'annotations')
        if (not self.force_download) and os.path.exists(path) and \
                os.path.exists(images_folder) :
            print(utils.log_color('\nINFO', 'dataset exists - will reuse', path))
            return
        #
        
        print(utils.log_color('\nINFO', 'downloading and preparing dataset', path + ' This may take some time.'))
        print(f'{Fore.YELLOW}'
              f'\nPoseCNN:'
              f'\n   A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes, RSS 2018, '
              f'\n         Xiang et al. https://arxiv.org/abs/1711.00199\n'
              f'\n    Visit the following urls to know more about the YCBV dataset. '
              f'\n        https://rse-lab.cs.washington.edu/projects/posecnn/ '
              f'\n        https://bop.felk.cvut.cz/datasets/ '
              f'\n        https://www.ycbbenchmarks.com/ '
              f'{Fore.RESET}\n')

        dataset_url = 'https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_test_bop19.zip'  # test_split
        base_url = 'https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_base.zip'   # base url
        object_model_url = "https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_models.zip"   # object model url
        download_root = os.path.join(root, 'download')
        base_path = utils.download_file(base_url, root=download_root, extract_root=os.path.dirname(root))
        dataset_path = utils.download_file(dataset_url, root=download_root, extract_root=root)
        object_model_path = utils.download_file(object_model_url, root=download_root, extract_root=root)
        print(utils.log_color('\nINFO', 'dataset ready', path))
        return

    def _get_root(self, path):
        path = path.rstrip('/')
        root = os.sep.join(os.path.split(path)[:-1])
        return root

    def ycb2coco(self, root_path, split='test'):
        """
        Convert YCB annotation to COCO format
        """
        annotations_dir = os.path.join(root_path, 'annotations')
        os.makedirs(annotations_dir, exist_ok=True)
        basepath = os.path.join(root_path, split)
        data_folders = sorted(os.listdir(basepath))
        outfile = os.path.join(annotations_dir, 'instances_{}.json'.format(split))

        for data_folder_idx, data_folder in enumerate(data_folders):
            data_path = os.path.join(basepath, data_folder)
            annotations_gt = dict()
            for f in ('scene_gt_info.json', 'scene_gt.json'):
                path = os.path.join(data_path,  f)
                if os.path.exists(path):
                    print("Loading {}".format(path))
                    with open(path) as foo:
                        annotations_gt[f.split('.')[0]] = json.load(foo)

            if data_folder_idx==0:
                coco = dict()
                coco["info"] = dict([("description", "YCBV dataset in COCO format"), ("url", "https://bop.felk.cvut.cz/media/data/bop_datasets/")])
                coco["images"] = []
                coco["type"] = "instance"
                coco["categories"] = []
                coco["annotations"] = []

                for obj_class in self.class_to_name:
                    category = dict([
                        ("supercategory", "object"),
                        ("id", obj_class),
                        ("name", self.class_to_name[obj_class])
                    ])

                    coco["categories"].append(category)
                obj_count = 0
                img_count = 0
            pbar = tqdm(enumerate(zip(list(annotations_gt['scene_gt'].items()), list(annotations_gt['scene_gt_info'].items()))), total=len(annotations_gt['scene_gt_info']))
            num_images = len(list(annotations_gt['scene_gt'].items()))
            for image_index, objects in pbar:
                objects_gt, objects_gt_info = objects[0], objects[1]
                filename = "{:06}".format(int(objects_gt[0])) + '.png'
                height, width = cv2.imread(data_path + '/rgb/' + filename).shape[:2]
                image = dict([
                    ("image_folder", data_folder),
                    ("id", img_count), #
                    ("file_name", filename),
                    ("height", height),
                    ("width", width),
                ])
                coco["images"].append(image)
                for object_gt, object_gt_info  in zip(objects_gt[1], objects_gt_info[1]):
                    if object_gt_info['visib_fract'] > 0:
                        annotation = dict([
                            ("image_id", img_count),
                            ("id", obj_count),
                            ("bbox", object_gt_info["bbox_obj"]),
                            ("area", object_gt_info["bbox_obj"][2] * object_gt_info["bbox_obj"][3]),
                            ("iscrowd", 0),
                            ("category_id", object_gt["obj_id"]-1),
                            ("R", object_gt["cam_R_m2c"]),
                            ("T", object_gt["cam_t_m2c"])
                        ])
                        obj_count += 1
                        coco["annotations"].append(annotation)
                img_count += 1
            pbar.close()
        json.dump(coco, outfile)



    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = self.coco_dataset.loadImgs([img_id])[0]
        image_path = os.path.join(self.image_dir,  img['image_folder'], 'rgb', img['file_name'])
        return image_path

    def __len__(self):
        return self.num_frames

    def __del__(self):
        for t in self.tempfiles:
            t.cleanup()
        #

    def __call__(self, predictions, **kwargs):
        return self.evaluate(predictions, **kwargs)

    def evaluate(self, preds, **kwargs):
        data_list = self.convert_to_coco_format(preds)
        eval_results_6dpose = self.evaluate_prediction_6dpose(data_list)
        accuracy = {'accuracy_add(s)_p1%': eval_results_6dpose[0]['ADD_0p1_avg']*100.0, 'accuracy_add(s)_p2%': eval_results_6dpose[0]['ADD_0p2_avg']*100.0}
        return accuracy

    def convert_to_coco_format(self, preds):
        data_list = []
        targets = self.coco_dataset.imgToAnns
        for frame_index, frame_targets in targets.items():
            if frame_index == len(preds):
                break
            frame_data_list = []
            pred = preds[frame_index]
            cls_pred = np.array(pred['cls'], dtype=np.int32)
            for target in frame_targets:
                pred_gt_data = {
                    "image_id": target['image_id'],
                    "bbox_gt": target['bbox'],
                    "rotation_gt": target['R'],
                    "translation_gt": target['T'],
                    "category_id": target['category_id'],
                    "missing_det": True
                }
                if np.sum(cls_pred == target['category_id']) > 0:
                    matched_index =  np.where(cls_pred == target['category_id'])[0][0]
                    pred_gt_data.update({
                        "bbox_pred": pred['bbox'][matched_index],
                        "score": pred['scores'][matched_index],
                        "segmentation": [],
                        "rotation_pred" : pred['rotation'][matched_index],
                        "translation_pred": pred['translation'][matched_index],
                        "missing_det": False
                    })
                frame_data_list.append(pred_gt_data)
            data_list.extend(frame_data_list)
        return data_list

    def evaluate_prediction_6dpose(self, data_dict):
        data_dict_asym = []
        data_dict_sym = []
        for pred_data in data_dict:
            if self.class_to_name[pred_data['category_id']] not in self.symmetric_objects.values():
                data_dict_asym.extend([pred_data])
            else:
                data_dict_sym.extend([pred_data])
        score_dict_asym = self.compute_add_score(data_dict_asym)
        score_dict_sym = self.compute_adds_score(data_dict_sym)
        score_dict = {}
        for metric in score_dict_asym.keys():
            score_dict[metric] = {**score_dict_asym[metric], **score_dict_sym[metric]}
            score_dict[metric + "_avg"] = np.mean(list(score_dict[metric].values()))

        score_dict_summmary = ""
        for metric in score_dict.keys():
            score_dict_summmary += metric + "\n"
            if "avg" not in metric:
                for cls in score_dict[metric].keys():
                    score_dict_summmary += "{:>5}".format(cls) + " ({:>12}) : ".format(self.class_to_name[cls]) + "{0:2f}".format(score_dict[metric][cls]) + "\n"
            else:
                score_dict_summmary += "{0:2f}".format(score_dict[metric]) + "\n"
        return score_dict, score_dict_summmary


    def compute_add_score(self, data_dict, percentage=0.1):
        distance_category = np.zeros((len(data_dict), 2))
        for index, pred_data in enumerate(data_dict):
            if not pred_data['missing_det']:
                R_gt, t_gt = np.array(pred_data['rotation_gt']).reshape(3,3), np.array(pred_data['translation_gt'])
                R_pred, t_pred= pred_data['rotation_pred'].reshape(3,3), pred_data['translation_pred']
                pts3d = self.class_to_model[pred_data['category_id']]
                #mean_distances = np.zeros((count,), dtype=np.float32)
                pts_xformed_gt = np.matmul(R_gt,  pts3d.transpose()) + t_gt[:, None]
                pts_xformed_pred = np.matmul(R_pred, pts3d.transpose()) + t_pred[:, None]
                distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
                distance_category[index, 0] = np.mean(distance)
                distance_category[index, 1] = pred_data['category_id']
            else:
                distance_category[index, 0] = 1e6 #This distance is set to a very high value for a missing detection.
                distance_category[index, 1] = pred_data['category_id']

        threshold = [self.class_to_diameter[category] * percentage for category in self.class_to_diameter.keys()]
        score_dict = {}
        for threshold_multiple in range(1, 6):
            score_dict["ADD_0p{}".format(threshold_multiple)] = {}
        for category_id, category_type in self.class_to_name.items():
            num_instances = len(distance_category[distance_category[:, 1] == category_id][:,0])
            if num_instances > 0:
                for threshold_multiple in range(1,6):
                    score = np.sum(distance_category[distance_category[:, 1] == category_id][:,0] < threshold_multiple * threshold[category_id]) / (num_instances + 1e-6)
                    score_dict["ADD_0p{}".format(threshold_multiple)].update({category_id: score})
        return score_dict


    def compute_adds_score(self, data_dict, percentage=0.1):
        distance_category = np.zeros((len(data_dict), 2))
        for index, pred_data in enumerate(data_dict):
            if not pred_data['missing_det']:
                R_gt, t_gt = np.array(pred_data['rotation_gt']).reshape(3,3), np.array(pred_data['translation_gt'])
                R_pred, t_pred = pred_data['rotation_pred'].reshape(3,3), pred_data['translation_pred']
                pts3d = self.class_to_model[pred_data['category_id']]
                pts_xformed_gt = np.matmul(R_gt, pts3d.transpose()) + t_gt[:, None]
                pts_xformed_pred = np.matmul(R_pred, pts3d.transpose()) + t_pred[:, None]
                kdt = KDTree(pts_xformed_gt.transpose(), metric='euclidean')
                distance, _ = kdt.query(pts_xformed_pred.transpose(), k=1)
                distance_category[index, 0] = np.mean(distance)
                distance_category[index, 1] = pred_data['category_id']
            else:
                distance_category[index, 0] = 1e6  #This distance is set to a very high value for a missing detection.
                distance_category[index, 1] = pred_data['category_id']

        threshold = [self.class_to_diameter[category] * percentage for category in self.class_to_diameter.keys()]
        score_dict = {}
        for threshold_multiple in range(1, 6):
            score_dict["ADD_0p{}".format(threshold_multiple)] = {}

        for category_id, category_type in self.class_to_name.items():
            num_instances = len(distance_category[distance_category[:, 1] == category_id][:, 0])
            if num_instances > 0:
                for threshold_multiple in range(1,6):
                    score = np.sum(distance_category[distance_category[:, 1] == category_id][:,0] < threshold_multiple * threshold[category_id]) / (num_instances + 1e-6)
                    score_dict["ADD_0p{}".format(threshold_multiple)].update({category_id: score})
        return score_dict



class CADModelsYCB():
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.cad_models_path = os.path.join(self.data_dir, "models_eval")
        self.class_to_name = {
                    0: "002_master_chef_can" , 1: "003_cracker_box" ,  2: "004_sugar_box" , 3: "005_tomato_soup_can",  4: "006_mustard_bottle",
                    5: "007_tuna_fish_can",  6: "008_pudding_box" , 7: "009_gelatin_box", 8: "010_potted_meat_can",  9: "011_banana",
                    10: "019_pitcher_base", 11: "021_bleach_cleanser",  12: "024_bowl", 13: "025_mug", 14: "035_power_drill",
                    15: "036_wood_block", 16: "037_scissors", 17: "040_large_marker", 18: "051_large_clamp", 19: "052_extra_large_clamp",
                    20: "061_foam_brick"
                    }

        self.models_dict_path = os.path.join(self.cad_models_path, "models_info.json")
        with open(self.models_dict_path) as foo:
            self.models_dict  = json.load(foo)
        self.class_to_model = self.load_cad_models()
        self.class_to_sparse_model = self.create_sparse_models()
        self.models_corners, self.models_diameter = self.get_models_params()
        self.camera_matrix = self.get_camera_params()
        self.symmetric_objects = { 1 : "002_master_chef_can" , 13 : "024_bowl" , 14 : "025_mug", 16 : "036_wood_block",
                                    18 : "040_large_marker", 19 : "051_large_clamp", 20 : "052_extra_large_clamp", 21 : "061_foam_brick" }

    def get_camera_params(self):
        camera_params_paths = [os.path.join(self.data_dir, "camera_uw.json"),
                                    os.path.join(self.data_dir, "camera_cmu.json")]
        camera_matrix = {}
        for camera_param_path in camera_params_paths:
            with open(camera_param_path) as foo:
                camera_params = json.load(foo)
            camera_name = os.path.basename(camera_param_path)
            camera_matrix[camera_name.split(".")[0]] = \
                np.array([camera_params['fx'], 0, camera_params['cx'], 0.0, camera_params['fy'], camera_params['cy'], 0.0, 0.0, 1.0])
        return camera_matrix


    def load_cad_models(self):
        class_to_model = {class_id: None for class_id in self.class_to_name.keys()}
        for class_id, name in self.class_to_name.items():
            file = "obj_{:06}.ply".format(class_id + 1)
            cad_model_path = os.path.join(self.cad_models_path, file)
            assert os.path.isfile(cad_model_path), "The file {} model for class {} was not found".format(file, name)
            class_to_model[class_id] = self.load_model_point_cloud(cad_model_path)
        return class_to_model



    def load_model_point_cloud(self, datapath):
        model = PlyData.read(datapath)
        vertex = model['vertex']
        points = np.stack([vertex[:]['x'], vertex[:]['y'], vertex[:]['z']], axis=-1).astype(np.float64)
        return points

    def get_models_params(self):
        """
        Convert model corners from (min_x, min_y, min_z, size_x, size_y, size_z) to actual coordinates format of dimension (8,3)
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
            models_corners_3d.update({int(model_id)-1: corners_3d})
            models_diameter.update({int(model_id)-1: model_param['diameter']})
        return models_corners_3d, models_diameter

    def create_sparse_models(self):
        class_to_sparse_model = {}
        for model_id in self.class_to_model.keys():
            sample_rate =len(self.class_to_model[model_id])//500
            #sparsely sample the model to have close to 500 points
            class_to_sparse_model.update({model_id : self.class_to_model[model_id][::sample_rate, :]})
        return class_to_sparse_model


