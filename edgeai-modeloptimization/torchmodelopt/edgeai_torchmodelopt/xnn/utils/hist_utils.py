#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
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

import sys
import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt

#from .. import layers as xtensor_layers

#study histogram for 3D tensor. The 1st dim belongs to ch
#study 2D histogram of each channel
def comp_hist_tensor3d(x=[], name='tensor', en=True, dir = 'dir_name', log = False, ch_dim=2):
    if en == False:
        return
    root = os.getcwd()
    path = root + '/checkpoints/debug/'+ dir
    if not os.path.exists(path):
        os.makedirs(path)

    if ch_dim == 2:
        for ch in range(x.shape[ch_dim]):
            x_ch = x[:,:,ch]
            print('min={}, max={}, std={}, mean={}'.format(x_ch.min(), x_ch.max(), x_ch.std(), x_ch.mean()))
            plt.hist(x_ch.flatten(), bins=256, log=log)  # arguments are passed to np.histogram
            #plt.title("Histogram with 'auto' bins")
            #plt.show()
            plt.savefig('{}/{}_{:03d}.jpg'.format(path, name, ch))
            plt.close()
    elif ch_dim == 0:
        for ch in range(x.shape[ch_dim]):
            x_ch = x[ch,:,:]
            print('min={}, max={}, std={}, mean={}'.format(x_ch.min(), x_ch.max(), x_ch.std(), x_ch.mean()))
            plt.hist(x_ch.flatten(), bins=256, log=log)  # arguments are passed to np.histogram
            #plt.title("Histogram with 'auto' bins")
            #plt.show()
            plt.savefig('{}/{}_{:03d}.jpg'.format(path, name, ch))
            plt.close()

def hist_tensor2D(x_ch=[], dir = 'tensor_dir', name='tensor', en=True, log=False, ch=0):
    if en == False:
        return
    root = os.getcwd()
    path = root + '/checkpoints/debug/'+ dir
    if not os.path.exists(path):
        os.makedirs(path)
    
    print('min={:.3f}, max={:.3f}, std={:.3f}, mean={:.3f}'.format(x_ch.min(), x_ch.max(), x_ch.std(), x_ch.mean()))
    hist_ary = plt.hist(x_ch.flatten(), bins=256, log=log)  # arguments are passed to np.histogram
    

    #plt.title("Histogram with 'auto' bins")
    #plt.show()
    plt.savefig('{}/{}_{:03d}.jpg'.format(path, name,ch))
    plt.close()
    return hist_ary

def analyze_model(model):
    num_dead_ch = 0
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if m.weight.shape[1] == 1:
                for ch in range(m.weight.shape[0]):
                    cur_ch_wt = m.weight[ch][0][...]
                    mn = cur_ch_wt.min()
                    mx = cur_ch_wt.max()
                    mn = mn.cpu().detach().numpy()
                    mx = mx.cpu().detach().numpy()
                    print(n, 'dws weight ch mn mx', ch, mn, mx)
                    #print(n, 'dws weight ch', ch, cur_ch_wt)
                    if max(abs(mn), abs(mx)) <= 1E-40:
                        num_dead_ch += 1
            else:
                print(n, 'weight', 'shape', m.weight.shape, m.weight.min(), m.weight.max())
                if m.bias is not None:
                    print(n, 'bias', m.bias.min(), m.bias.max())        

    print("num_dead_ch: ", num_dead_ch)                

def study_wts(self, modules):
    for key, value in modules.items():
        print(key, value)
        for key2, value2 in value._modules.items():
            print(key2, value2)
            print(value2.weight.shape)


def comp_hist(self, x=[], ch_idx=0, name='tensor'):
    #hist_pred = torch.histc(x.cpu(), bins=256)
    root = os.getcwd()
    path = root + '/checkpoints/debug/'+ name
    if not os.path.exists(path):
        os.makedirs(path)

    #print(hist_pred)
    for ch in range(x.shape[1]):
        x_ch = x[0][ch]
        plt.hist(x_ch.view(-1).cpu().numpy(), bins=256)  # arguments are passed to np.histogram
        print('min={}, max={}, std={}, mean={}'.format(x_ch.min(), x_ch.max(), x_ch.std(), x_ch.mean()))
        #plt.title("Histogram with 'auto' bins")
        #plt.show()
        plt.savefig('{}/{}_{:03d}.jpg'.format(path, name, ch))
        plt.close()


def store_layer_op(en=False, tensor= [], name='tensor_name'):
    if en == False:
        return

    # write tensor
    tensor = tensor.astype(np.int16)
    print("writing tensor {} : {} : {} : {} : {}".format(name, tensor.shape, tensor.dtype, tensor.min(), tensor.max()))

    root = os.getcwd()
    tensor_dir = root + '/checkpoints/debug/' + name

    if not os.path.exists(tensor_dir):
        os.makedirs(tensor_dir)

    tensor_name = tensor_dir + "{}.npy".format(name)
    np.save(tensor_name, tensor)
    comp_hist_tensor3d(x=tensor, name=name, en=True, dir=name, log=True, ch_dim=0)