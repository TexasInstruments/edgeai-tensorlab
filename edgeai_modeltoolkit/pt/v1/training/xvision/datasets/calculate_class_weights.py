#################################################################################
# Copyright (c) 2018-2021, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################
# Some parts of the code are borrowed from: https://github.com/ansleliu/LightNet
# with the following license:
#
# MIT License
#
# Copyright (c) 2018 Huijun Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import os
import scipy.misc as misc
import sys

from .cityscapes_plus import CityscapesBaseSegmentationLoader, CityscapesBaseMotionLoader


def calc_median_frequency(classes, present_num):
    """
    Class balancing by median frequency balancing method.
    Reference: https://arxiv.org/pdf/1411.4734.pdf
       'a = median_freq / freq(c) where freq(c) is the number of pixels
        of class c divided by the total number of pixels in images where
        c is present, and median_freq is the median of these frequencies.'
    """
    class_freq = classes / present_num
    median_freq = np.median(class_freq)
    return median_freq / class_freq


def calc_log_frequency(classes, value=1.02):
    """Class balancing by ERFNet method.
       prob = each_sum_pixel / each_sum_pixel.max()
       a = 1 / (log(1.02 + prob)).
    """
    class_freq = classes / classes.sum()  # ERFNet is max, but ERFNet is sum
    # print(class_freq)
    # print(np.log(value + class_freq))
    return 1 / np.log(value + class_freq)


def calc_weights():
    method = "median"
    result_path = "/afs/cg.cs.tu-bs.de/home/zhang/SEDPShuffleNet/datasets"

    traval = "gtFine"
    imgs_path = "./data/tiad/data/leftImg8bit/train"    #"./data/cityscapes/data/leftImg8bit/train"   #"./data/TIAD/data/leftImg8bit/train"
    lbls_path = "./data/tiad/data/gtFine/train"         #"./data/cityscapes/data/gtFine/train"   # "./data/tiad/data/gtFine/train"  #"./data/cityscapes_frame_pair/data/gtFine/train"
    labels = xnn.utils.recursive_glob(rootdir=lbls_path, suffix='labelTrainIds_motion.png')  #'labelTrainIds_motion.png'  #'labelTrainIds.png'

    num_classes = 2       #5  #2

    local_path = "./data/checkpoints"
    dst = CityscapesBaseMotionLoader() #TiadBaseSegmentationLoader()  #CityscapesBaseSegmentationLoader()  #CityscapesBaseMotionLoader()

    classes, present_num = ([0 for i in range(num_classes)] for i in range(2))

    for idx, lbl_path in enumerate(labels):
        lbl = misc.imread(lbl_path)
        lbl = dst.encode_segmap(np.array(lbl, dtype=np.uint8))

        for nc in range(num_classes):
            num_pixel = (lbl == nc).sum()
            if num_pixel:
                classes[nc] += num_pixel
                present_num[nc] += 1

    if 0 in classes:
        raise Exception("Some classes are not found")

    classes = np.array(classes, dtype="f")
    presetn_num = np.array(classes, dtype="f")
    if method == "median":
        class_weight = calc_median_frequency(classes, present_num)
    elif method == "log":
        class_weight = calc_log_frequency(classes)
    else:
        raise Exception("Please assign method to 'mean' or 'log'")

    print("class weight", class_weight)
    print("Done!")


if __name__ == '__main__':
    calc_weights()