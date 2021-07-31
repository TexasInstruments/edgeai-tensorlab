import os
import glob
import re


dest_url = 'http://software-dl.ti.com/jacinto7/esd/modelzoo/common/models/'

supported_ext = ['.tf', '.tflite', '.pth', '.pt', '.ptl', '.onnx', '.json', '.params', '.prototxt', '.caffemodel']

files = glob.glob('models/*/*/*/*/*')

for file_name in files:
    file_ext = os.path.splitext(file_name)[1]
    if file_ext in supported_ext:
        dirname = os.path.dirname(file_name)
        basename = os.path.basename(file_name)
        dest_filename = re.sub('^models/', dest_url, file_name)
        #print(file_name, new_filename)
        link_name = file_name + '.link'
        print(link_name)
        with open(link_name, 'w') as fp:
            fp.write(dest_filename)
        #
