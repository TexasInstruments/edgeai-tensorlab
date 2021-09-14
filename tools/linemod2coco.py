import os
from typing import OrderedDict
from cv2 import data
import mmcv
from PIL import Image
from numpy import true_divide
import yaml
import shutil

def convert_to_coco_json(gt_dict, test_list, dataset_path, outfile_train, outfile_test):
    image_id = 0

    test_func = True

    coco_train = dict()
    coco_train["images"] = []
    coco_train["type"] = "instance"
    coco_train["categories"] = []
    coco_train["annotations"] = []

    coco_test = dict()
    coco_test["images"] = []
    coco_test["type"] = "instance"
    coco_test["categories"] = []
    coco_test["annotations"] = []

    image_set = set()

    class_to_name = {0: "ape", 1: "benchvise", 2: "bowl", 3: "can", 4: "cat", 5: "cup", 6: "driller", 7: "duck", 8: "glue", 9: "holepuncher", 10: "iron", 11: "lamp", 12: "phone", 13: "cam", 14: "eggbox"}
    for obj_class in class_to_name:
        category = dict([
            ("supercategory", "object"),
            ("id", obj_class),
            ("name", class_to_name[obj_class])
        ])

        coco_train["categories"].append(category)
        coco_test["categories"].append(category)
    
    obj_count = 0
    prev_image_num = -1
    for image_num, objects in gt_dict.items():
        for object in objects:
            
            filename = ''
            for i in range(4 - digits(image_num)):
                filename += '0'
            filename += (str(image_num) + '.png')

            height, width = mmcv.imread(dataset_path + '/rgb/' + filename).shape[:2]
            image = dict([
                ("id", image_num),
                ("file_name", filename),
                ("height", height),
                ("width", width)
            ])

            annotation = dict([
                ("image_id", image_num),
                ("id", obj_count),
                ("bbox", object["obj_bb"]),
                ("area", object["obj_bb"][2] * object["obj_bb"][3]),
                ("iscrowd", 0),
                ("category_id", object["obj_id"]),
                ("R", object["cam_R_m2c"]),
                ("T", object["cam_t_m2c"])
            ])

            if test_func:
                print(annotation)
                test_func = False

            print("Completed image number", image_num)
            obj_count += 1

            if filename[:-4] in test_list:
                if prev_image_num != image_num:
                    coco_test["images"].append(image)
                    prev_image_num = image_num
                coco_test["annotations"].append(annotation)
            else:
                if prev_image_num != image_num:
                    coco_train["images"].append(image)
                    prev_image_num = image_num
                coco_train["annotations"].append(annotation)

    mmcv.dump(coco_train, outfile_train)
    mmcv.dump(coco_test, outfile_test)

def sort_images(src, train_dst, test_dst, test_list):
    for image_num in range(1214):
        filename = ''
        for i in range(4 - digits(image_num)):
            filename += '0'
        filename += (str(image_num) + '.png')

        if filename[:-4] in test_list:
            shutil.copy(os.path.join(src,filename), os.path.join(test_dst, filename))
        else:
            shutil.copy(os.path.join(src,filename), os.path.join(train_dst, filename))

def digits(num):
    if num == 0:
        return 1

    count = 0
    while num != 0:
        num = int(num/10)
        count += 1

    return count

if __name__ == "__main__":
    datapath = '/home/a0492969/datasets/Linemod_preprocessed/data/02'
    src_datapath = '/home/a0492969/datasets/Linemod_preprocessed/data/02/rgb'
    train_dst_datapath = '/home/a0492969/datasets/Occlusion_COCO/train'
    test_dst_datapath = '/home/a0492969/datasets/Occlusion_COCO/test'
    gt_dict = dict()
    test_list = []
    train_list = []
    with open(datapath + '/gt.yml') as yaml_file:
        gt_dict = yaml.safe_load(yaml_file)
    
    with open(datapath + '/train.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            test_list.append(l.replace("\n",""))

    convert_to_coco_json(gt_dict, test_list, datapath, 'instances_train.json', 'instances_test.json')
    #sort_images(src_datapath, train_dst_datapath, test_dst_datapath,  test_list)
