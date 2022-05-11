import os
import mmcv
import yaml
import shutil
from tqdm import tqdm

def convert_to_coco_json(merge=False):

    basepath = '/data/ssd/6d_pose/Linemod_preprocessed/data/'
    data_folders = sorted(os.listdir(basepath))

    if merge:
        outfile_train_merged = '/data/ssd/6d_pose/LINEMOD_COCO/instances_train.json'
        outfile_test_merged = '/data/ssd/6d_pose/LINEMOD_COCO/instances_test.json'

    for data_folder_idx, data_folder in enumerate(data_folders):
        data_path = os.path.join(basepath, data_folder)
        print("Loading {}".format(data_path + '/gt.yml'))
        with open(data_path + '/gt.yml') as yaml_file:
            gt_dict = yaml.safe_load(yaml_file)
        print("Loading completed {}".format(data_path + '/gt.yml'))
        with open(data_path + '/train.txt', 'r') as f:
            test_list = list(f)
            test_list = [idx.rstrip() for idx in test_list]

        if not merge or data_folder_idx==0:
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

        pbar = tqdm(enumerate(gt_dict.items()), total=len(gt_dict))
        for image_num, objects in pbar:
            for object in objects[1]:
                filename = "{:04}".format(image_num) + '.png'
                height, width = mmcv.imread(data_path + '/rgb/' + filename).shape[:2]
                image = dict([
                    ("image_folder", data_folder),
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
        pbar.close()
        if not merge:
            outfile_train = '/data/ssd/6d_pose/LINEMOD_COCO/instances_train_{}.json'.format(data_folder)
            outfile_test = '/data/ssd/6d_pose/LINEMOD_COCO/instances_test_{}.json'.format(data_folder)
            mmcv.dump(coco_train, outfile_train)
            mmcv.dump(coco_test, outfile_test)

    if merge:
        mmcv.dump(coco_train, outfile_train_merged)
        mmcv.dump(coco_test, outfile_test_merged)


def sort_images(src, train_dst, test_dst, test_list):
    for image_num in range(1214):
        filename = "{:04}".format(image_num) + '.png'
        if filename[:-4] in test_list:
            shutil.copy(os.path.join(src,filename), os.path.join(test_dst, filename))
        else:
            shutil.copy(os.path.join(src,filename), os.path.join(train_dst, filename))


if __name__ == "__main__":
        convert_to_coco_json(merge=True)
