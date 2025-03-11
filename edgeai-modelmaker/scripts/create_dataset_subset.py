import os
import shutil
import json

with open('annotations/instances.json') as fp:
    annotations = json.load(fp)

images = annotations['images']

images_file_list = os.listdir('images')
polygons_file_list = os.listdir('polygons')

os.makedirs('images_selected')
os.makedirs('polygons_selected')

for image in images:
    file_name = image['file_name']
    print(file_name)
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    annotated_image_files = [f for f in images_file_list if base_name in f]
    annotated_polygon_files = [f for f in polygons_file_list if base_name in f]
    #
    shutil.copy2(os.path.join('images', file_name), os.path.join('images_selected', file_name))
    for f in annotated_polygon_files:
        shutil.copy2(os.path.join('polygons', f), os.path.join('polygons_selected', f))
    #
