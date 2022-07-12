import os
import mmcv
import yaml
import shutil
from tqdm import tqdm
import json

def convert_to_coco_json(merge=False):
    basepath = '/data/ssd/6d_pose/LINEMO_Synthetic/train_pbr'
    data_folders = sorted(os.listdir(basepath))

    if merge:
        outfile_train_merged = '/data/ssd/6d_pose/LINEMOD_Occlusion_COCO_pbr/instances_train.json'

    for data_folder_idx, data_folder in enumerate(data_folders):
        data_path = os.path.join(basepath, data_folder)
        annotations_gt = dict()
        for f in ('scene_gt_info.json', 'scene_gt.json'):
            path = os.path.join(data_path,  f)
            if os.path.exists(path):
                print("Loading {}".format(path))
                with open(path) as foo:
                    annotations_gt[f.split('.')[0]] = json.load(foo)

        if not merge or data_folder_idx==0:
            coco_train = dict()
            coco_train["images"] = []
            coco_train["type"] = "instance"
            coco_train["categories"] = []
            coco_train["annotations"] = []

            class_to_name = {1: "ape", 2: "benchvise", 3: "bowl", 4: "can", 5: "cat", 6: "cup", 7: "driller", 8: "duck", 9: "glue", 10: "holepuncher", 11: "iron", 12: "lamp", 13: "phone", 14: "cam", 15: "eggbox"}
            for obj_class in class_to_name:
                category = dict([
                    ("supercategory", "object"),
                    ("id", obj_class),
                    ("name", class_to_name[obj_class])
                ])

                coco_train["categories"].append(category)

            obj_count = 0

        pbar = tqdm(enumerate(zip(list(annotations_gt['scene_gt'].items()), list(annotations_gt['scene_gt_info'].items()))), total=len(annotations_gt['scene_gt_info']))
        num_images = len(list(annotations_gt['scene_gt'].items()))
        for image_index, objects in pbar:
            objects_gt, objects_gt_info = objects[0], objects[1]
            filename = "{:06}".format(image_index) + '.jpg'
            height, width = mmcv.imread(data_path + '/rgb/' + filename).shape[:2]
            image = dict([
                ("image_folder", data_folder),
                ("id", image_index+num_images*data_folder_idx), #
                ("file_name", filename),
                ("height", height),
                ("width", width),
                ("type", 'synthetic')
            ])
            coco_train["images"].append(image)
            for object_gt, object_gt_info  in zip(objects_gt[1], objects_gt_info[1]):
                if object_gt_info['visib_fract'] > 0:
                    annotation = dict([
                        ("image_id", image_index+num_images*data_folder_idx),
                        ("id", obj_count),
                        ("bbox", object_gt_info["bbox_obj"]),
                        ("area", object_gt_info["bbox_obj"][2] * object_gt_info["bbox_obj"][3]),
                        ("iscrowd", 0),
                        ("category_id", object_gt["obj_id"]),
                        ("R", object_gt["cam_R_m2c"]),
                        ("T", object_gt["cam_t_m2c"])
                    ])
                    obj_count += 1
                    coco_train["annotations"].append(annotation)
        pbar.close()
        if not merge:
            outfile_train = '/data/ssd/6d_pose/LINEMOD_COCO/instances_train_{}.json'.format(data_folder)
            mmcv.dump(coco_train, outfile_train)


    merge_linemod_real = True #
    if merge_linemod_real:
        linemod_real_path = "/data/ssd/6d_pose/LINEMOD_Occlusion_COCO/annotations/instances_train.json"
        with open(linemod_real_path) as foo:
            linemod_real_gt = json.load(foo)
            print("loading real LINEMOD gt data")
            for image_dict in linemod_real_gt["images"]:
                image_dict['id'] = image_dict['id'] + num_images * 50
                print("image_dict", image_dict['id'])

            for annotation_dict in linemod_real_gt["annotations"]:
                annotation_dict['image_id'] = annotation_dict['image_id'] + num_images*50 # num_folders=50. TO remove the hard-coding
                print("annotation_dict", annotation_dict['image_id'])

            coco_train["annotations"].extend(linemod_real_gt["annotations"])
            coco_train["images"].extend(linemod_real_gt["images"])

    if merge:
        mmcv.dump(coco_train, outfile_train_merged)

def sort_images(src, train_dst, test_dst, test_list):
    for image_num in range(1214):
        filename = "{:04}".format(image_num) + '.png'
        if filename[:-4] in test_list:
            shutil.copy(os.path.join(src,filename), os.path.join(test_dst, filename))
        else:
            shutil.copy(os.path.join(src,filename), os.path.join(train_dst, filename))


if __name__ == "__main__":
        convert_to_coco_json(merge=True)
