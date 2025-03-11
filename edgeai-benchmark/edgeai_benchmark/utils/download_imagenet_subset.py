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

#################################################################################
#This is the helper file to create subset dataset from larger dataset
#################################################################################
import os
import yaml
from pathlib import Path
#from jacinto_ai_benchmark import *
import jacinto_ai_benchmark.utils

#####################
#extract multiple tars and store them in separate folders
#####################
def extract_multiple_tar(data_root=None, extracted_path=None):
    for subdir, dirs, files in sorted(os.walk(data_root)):
        #print(subdir, dirs, files)
        p = Path(subdir)
        for file in sorted(files):
            filename, file_extension = os.path.splitext(file)
            if file_extension in ['.tar']:
                input_tar_file = os.path.join(subdir, file)
                op_path_one_set = os.path.join(extracted_path,filename)
                os.makedirs(op_path_one_set, exist_ok=True)
                cmd = "tar -xvf {} -C {}".format(input_tar_file, op_path_one_set)
                print(cmd)
                os.system(cmd)
    return            

#####################
#generate folder name to class index mapping
#####################
def gen_folder_to_index(train_labels=None, op_yaml_file = None):
    # Using readlines()
    file1 = open(train_labels, 'r')
    lines = file1.readlines()

    folder_id_to_label_id = dict()
    
    count = 0
    # Strips the newline character
    for line in lines:
        count += 1
        #print("Line{}: {}".format(count, line.strip()))
        #folder_id_to_label_id[]
        line = line.strip()
        dir_name = line.split(" ")[0]
        dir_name = dir_name.split("/")[0]

        label_idx = line.split(" ")[1]
        # print(dir_name)
        # print(label_idx)
        folder_id_to_label_id[dir_name] = label_idx

    folder_id_to_label_id = utils.pretty_object(folder_id_to_label_id)
    with open(os.path.join(op_yaml_file), 'w') as fp:
        yaml.safe_dump(folder_id_to_label_id, fp, sort_keys=False)
    
    return    

#####################
#generate validation text file with relative name
#####################
def gen_validation_txt_file(folder_id_to_label_id=None, op_val_text=None):

    #read folder name to index
    with open(folder_id_to_label_id) as fp:
        map_dict = yaml.safe_load(fp)

    f = open(op_val_text, 'w')
    for subdir, dirs, files in sorted(os.walk(extracted_path)):
        #print(subdir, dirs, files)
            for file in sorted(files):
                filename, file_extension = os.path.splitext(file)
                if file_extension in ['.JPEG']:
                    input_file = os.path.join(subdir, file)
                    image_file_path_relative = Path(input_file).relative_to(extracted_path)
                    dir_path_relative = Path(subdir).relative_to(extracted_path)
                    print(image_file_path_relative, " ", map_dict[str(dir_path_relative)], file=f)
                    print(image_file_path_relative, " ", map_dict[str(dir_path_relative)])
    return                

#####################
# Create val set with target accuracy.
# This is achieved by sampling from positive and negative images based on target accuracy
# Accuracy information of each image needed for this 
#####################
def create_val_set(params=None):
    targets = dict()
    # targets['target_accuracy'] = 75
    targets['num_images_per_class'] = params['num_images_target']//params['num_classes']
    targets['num_correct_images_per_class'] = (targets['num_images_per_class'] * params['target_accuracy'])//100
    targets['num_incorrect_images_per_class'] = targets['num_images_per_class'] - targets['num_correct_images_per_class'] 
    
    file1 = open(params['full_val_text'], 'r')
    lines = file1.readlines()

    class_stats = dict()
    count = 0
    #first identify how many classes are there.
    for line in lines:
        line = line.strip()
        class_idx = line.split(" ")[1]
        class_stats[class_idx] = dict()
        class_stats[class_idx]['correct_images'] = 0
        class_stats[class_idx]['incorrect_images'] = 0
        count += 1
      
    with open(params['accuracy_yaml']) as fp:
        accuracy_dict = yaml.safe_load(fp)

    op_val_text = os.path.join(params['op_val_text_path'], 'temp_val_file.txt')
    f = open(op_val_text, 'w')

    #sample positive and negative images for each class 
    count = 0
    nimages_in_new_set = 0
    for line in lines:
        line = line.strip()
        img_name = line.split(" ")[0]
        img_name = img_name.split("/")[-1]
        class_idx = line.split(" ")[1]
        #correct_image = accuracy_dict[img_name][0]
        correct_image = accuracy_dict[img_name]
        if (class_stats[class_idx]['correct_images'] < targets['num_correct_images_per_class']) and correct_image == 100.0:
            print(line, correct_image)
            print(line, file=f)
            class_stats[class_idx]['correct_images'] += 1
            nimages_in_new_set += 1

        if (class_stats[class_idx]['incorrect_images'] < targets['num_incorrect_images_per_class']) and correct_image == 0.0:
            print(line, correct_image)
            print(line, file=f)
            class_stats[class_idx]['incorrect_images'] += 1
            nimages_in_new_set += 1
        #get num_images_per_class file with class_idx
        count += 1
            
    filename = os.path.splitext(params['full_val_text'])[0]        
    op_val_text_final = "_".join(filename.split("_")[0:-1]) + '_' + str(nimages_in_new_set) + os.path.splitext(params['full_val_text'])[1]
    os.rename(op_val_text, op_val_text_final)
    print("Total images selected in the subset: ", nimages_in_new_set)    
    return
        
if __name__ == '__main__':        
    en_extract_tars = False
    if en_extract_tars:
        data_root = './dependencies/datasets/imagenet_class120/ILSVRC2012_img_train_t3_tars'
        extracted_path = './dependencies/datasets/imagenet_class120/ILSVRC2012_img_train_t3'
        extract_multiple_tar(data_root=data_root, extracted_path=extracted_path)
    
    #generate folder name to class index mapping
    en_gen_folder_to_index = False
    if en_gen_folder_to_index:
        train_labels = './dependencies/datasets/imagenet/train.txt'
        op_yaml_file = './dependencies/datasets/imagenet_class120/folder_id_to_label_id.yaml'
        gen_folder_to_index(train_labels=train_labels, op_yaml_file = op_yaml_file)
      
    en_gen_val_text_file = False
    if en_gen_val_text_file:
        folder_id_to_label_id = './dependencies/datasets/imagenet_class120/folder_id_to_label_id.yaml'
        op_val_text = './dependencies/datasets/imagenet_class120/imagenet_subset_val.txt'
        gen_validation_txt_file(folder_id_to_label_id=folder_id_to_label_id, op_val_text=op_val_text)

    en_create_val_set = True
    if en_create_val_set:
        params = dict()
        params['num_classes'] = 120
        params['target_accuracy'] = 75
        params['num_images_target'] = 360
        params['accuracy_yaml'] = './dependencies/datasets/imagenet_class120/ILSVRC2012_img_train_t3_accuracy.yaml'
        params['full_val_text'] = './dependencies/datasets/imagenet_class120/ILSVRC2012_img_train_t3_20580.txt'
        params['op_val_text_path'] = './dependencies/datasets/imagenet_class120/'
    
    # create subset of validation set with target accuracy. 
    create_val_set(params=params)
