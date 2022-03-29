import os
import sys
import json
import glob
import PIL
import PIL.Image

source_dir = '.'
png_files = glob.glob(f'{source_dir}/*/*/*/*.png')

for png_file in png_files:
    jpg_file = f'{os.path.splitext(png_file)[0]}.jpg'
    print(png_file, jpg_file)
    png_img = PIL.Image.open(png_file)
    png_img = png_img.convert("RGB")
    png_img.save(jpg_file)

