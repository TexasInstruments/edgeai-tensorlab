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

"""
Reference: https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html

FlowNet: Learning Optical Flow with Convolutional Networks,
A. Dosovitskiy and P. Fischer and E. Ilg and P. H{\"a}usser and C. Haz{\i}rba{\c{s}} and V. Golkov and P. v.d. Smagt and D. Cremers and T. Brox,
IEEE International Conference on Computer Vision (ICCV), 2015,
http://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15"

Occlusions, Motion and Depth Boundaries with a Generic Network for Disparity, Optical Flow or Scene Flow Estimation
E. Ilg and T. Saikia and M. Keuper and T. Brox
European Conference on Computer Vision (ECCV), 2018,
http://lmb.informatik.uni-freiburg.de/Publications/2018/ISKB18
"""


import os.path
import glob
from .dataset_utils import split2list, ListDataset

__all__ = ['flying_chairs']


def flying_chairs(dataset_config, root, transforms=None, split=None):
    train_list, test_list = make_dataset(root,split)
    train_dataset = ListDataset(root, train_list, transforms[0])
    test_dataset = ListDataset(root, test_list, transforms[1])
    return train_dataset, test_dataset


############################################################################
# internal functions
############################################################################

def make_dataset(dir, split=None):
    '''Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm    [name]_flow.flo' '''
    images = []
    for flow_map in glob.iglob(os.path.join(dir,'*_flow.flo')):
        flow_map = os.path.basename(flow_map)
        root_filename = flow_map[:-9]
        img1 = root_filename+'_img1.ppm'
        img2 = root_filename+'_img2.ppm'
        if not (os.path.isfile(os.path.join(dir,img1)) or os.path.isfile(os.path.join(dir,img2))):
            continue

        imgs = [img1,img2]
        flo = [flow_map]
        images.append((imgs,flo))

    return split2list(images, split)