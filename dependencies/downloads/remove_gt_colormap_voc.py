# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#Source: modified from https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/remove_gt_colormap.py

"""Removes the color map from segmentation annotations.

Removes the color map from the ground truth segmentation annotations and save
the results to output_dir.
"""
import glob
import os
import numpy as np
from PIL import Image
import argparse


def _remove_colormap(filename):
  """Removes the color map from the annotation.

  Args:
    filename: Ground truth annotation filename.

  Returns:
    Annotation without color map.
  """
  return np.array(Image.open(filename))


def _save_annotation(annotation, filename):
  """Saves the annotation as png file.

  Args:
    annotation: Segmentation annotation.
    filename: Output filename.
  """
  pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
  pil_image.save(filename)


def main(args):
  # Create the output directory if not exists.
  if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

  annotations = glob.glob(os.path.join(args.original_gt_folder,
                                       '*.' + args.segmentation_format))
  for annotation in annotations:
    raw_annotation = _remove_colormap(annotation)
    filename = os.path.basename(annotation)[:-4]
    _save_annotation(raw_annotation,
                     os.path.join(
                         args.output_dir,
                         filename + '.' + args.segmentation_format))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_gt_folder', type=str, default='./dependencies/downloads/VOCdevkit/VOC2012/SegmentationClass')
    parser.add_argument('--segmentation_format', type=str, default='png')
    parser.add_argument('--output_dir', type=str, default='./dependencies/downloads/VOCdevkit/VOC2012/SegmentationClassRaw')
    args = parser.parse_args()
    main(args)
