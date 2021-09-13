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

import torch
import numpy as np


##################################################################
def l2_norm(x,y=None):
    if y is not None:
        assert x.size() == y.size(), 'tensor dimension mismatch'
    #
    diff = (x-y) if y is not None else x
    return torch.norm(diff,p=2,dim=1,keepdim=True)

def l2_norm_self(x,y=None):
    return torch.norm(x,p=2,dim=1,keepdim=True)

def l1_norm(x,y=None):
    if y is not None:
        assert x.size() == y.size(), 'tensor dimension mismatch'
    #
    diff = (x-y) if y is not None else x
    return torch.norm(diff,p=1,dim=1,keepdim=True)

def smooth_l1_loss(x,y):
    return torch.nn.functional.smooth_l1_loss(x, y, reduction='none')

def abs_diff(x,y):
    assert x.size() == y.size(), 'tensor dimension mismatch'
    diff = x-y
    return torch.abs(diff)

def square_diff(x,y):
    assert x.size() == y.size(), 'tensor dimension mismatch'
    diff = x-y
    return (diff*diff)

def abs_relative_diff(x, y, eps = 0.0, max_val=None):
    assert x.size() == y.size(), 'tensor dimension mismatch'
    if max_val is not None:
        x = torch.clamp(x, -max_val, max_val)
        y = torch.clamp(y, -max_val, max_val)
    #

    diff = torch.abs(x - y)
    y = torch.abs(y)

    den_valid = (y == 0).float()
    eps_arr = (den_valid * (1e-6))   # Just to avoid divide by zero

    large_arr = (y > eps).float()    # ARD is not a good measure for small ref values. Avoid them.
    out = (diff / (y + eps_arr)) * large_arr
    return out

#give dif weight to target with max_value
def abs_relative_diff_weighted(x, y, eps = 0.0, max_val=None, weights=None):
    assert x.size() == y.size(), 'tensor dimension mismatch'
    if max_val is not None:
        x = torch.clamp(x, -max_val, max_val)
        y = torch.clamp(y, -max_val, max_val)
    #

    diff = torch.abs(x - y)
    y = torch.abs(y)

    den_valid = (y == 0).float()
    eps_arr = (den_valid * (1e-6))   # Just to avoid divide by zero

    large_arr = (y > eps).float()    # ARD is not a good measure for small ref values. Avoid them.
    out = (diff / (y + eps_arr)) * large_arr

    #print(torch.sum(out[y < max_val]))
    #print(torch.sum(out[y >= max_val]))
    if max_val is not None and weights is not None:
        out = out*(y<max_val).float()*weights[0] + out*(y>=max_val).float()*weights[1]
    #
    return out


def abs_relative_diff_x100(x, y, eps = 0.0, max_val=None):
    return abs_relative_diff(x, y, eps, max_val) * 100.0

def charbonnier(x, y, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    assert x.size() == y.size(), 'tensor dimension mismatch'
    diff = charbonnier_diff((x-y), mask, truncate, alpha, beta, epsilon)
    return diff


def charbonnier_diff(diff, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.01):
    epsilon2 = epsilon*epsilon
    offset = np.power(epsilon2, alpha)
    diff2 = (diff*beta)*(diff*beta)
    error = torch.pow(diff2 + epsilon2, alpha) - offset
    if mask is not None:
        error = mask*error
    if truncate is not None:
        error = torch.minimum(error, truncate)
    return error


def charbonnier_adaptive(x, y, mask=None, truncate=None, alpha=0.45, beta=1.0):
    assert x.size() == y.size(), 'tensor dimension mismatch'
    diff = charbonnier_adaptive_diff((x-y), mask, truncate, alpha, beta)
    return diff


def charbonnier_adaptive_diff(diff, mask=None, truncate=None, alpha=0.45, beta=1.0):
    epsilon = torch.median(diff)/2
    epsilon2 = epsilon*epsilon
    offset = torch.pow(epsilon2, alpha)
    diff2 = (diff*beta)*(diff*beta)
    error = torch.pow(diff2 + epsilon2, alpha) - offset
    if mask is not None:
        error = mask*error
    if truncate is not None:
        error = torch.minimum(error, truncate)
    return error


def berhu(x, y):
    assert x.size() == y.size(), 'tensor dimension mismatch'
    diff = torch.abs(x-y)
    epsilon = torch.median(diff)
    mask = ((diff<epsilon) == True)
    nmask = (mask == False)
    mask = mask.float()
    nmask = nmask.float()
    out = mask*diff + nmask*(diff*diff + epsilon*epsilon)/(2*epsilon)
    return out

##################################################################
def error_variance(x,y):
    assert x.size() == y.size(), 'tensor dimension mismatch'
    diff = x-y
    return torch.var(diff)


def error_std(x,y):
    assert x.size() == y.size(), 'tensor dimension mismatch'
    diff = x-y
    return torch.std(diff)


##################################################################
def gradient_func(pred):
    D_dy = pred[:, :, 1:] - pred[:, :, :-1]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy


def gradient_loss(in1, in2):
    dx1, dy1 = gradient_func(in1)
    dx2, dy2 = gradient_func(in2)
    loss = ((dx1-dx2).abs().mean()+(dy1-dy2).abs().mean())
    return loss


def smooth_loss(pred_disp):
    loss = 0
    dx, dy = gradient_func(pred_disp)
    dx2, dxdy = gradient_func(dx)
    dydx, dy2 = gradient_func(dy)
    loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())
    return loss


# second order smoothness constraint
# https://arxiv.org/abs/1711.07837
def smooth_loss2(pred, smooth_weight_h=1.0, smooth_weight_v=1.0, smooth_weight_d=1.0):
    if (pred.shape[2] < 4) or (pred.shape[3] < 4):
        return 0.0

    #D_dy = pred[..., :-2, :]   + pred[..., 2:, :],   2*pred[..., 1:-1, :]
    #D_dx = pred[..., :, :-2]   + pred[..., :, 2:],   2*pred[..., :, 1:-1]
    #D_d1 = pred[..., :-2, :-2] + pred[..., 2:, 2:],  2*pred[..., 1:-1, 1:-1]
    #D_d2 = pred[..., :-2, 2:]  + pred[..., 2:, :-2], 2*pred[..., 1:-1, 1:-1]
    #sloss = charbonnier_diff(D_dx + D_dy + D_d1 + D_d2).mean()

    weights_array = np.zeros((4, pred.size(1), 3, 3))
    # np.array conversion below is not required,
    # it is only done to ease the scaling.
    filter_x = np.array([[0,0,0],[1,-2,1],[0,0,0]])*smooth_weight_h
    filter_y = np.array([[0,1,0],[0,-2,0],[0,1,0]])*smooth_weight_v
    filter_d1 = np.array([[1,0,0],[0,-2,0],[0,0,1]])*smooth_weight_d
    filter_d2 = np.array([[0,0,1],[0,-2,0],[1,0,0]])*smooth_weight_d
    for ch in range(pred.size(1)):
        weights_array[0,ch,...] = filter_x
        weights_array[1,ch,...] = filter_y
        weights_array[2,ch,...] = filter_d1
        weights_array[3,ch,...] = filter_d2

    weights_tensor = torch.from_numpy(weights_array).float()
    weights_var = torch.Tensor(weights_tensor, requires_grad=False).float().cuda()
    grad = torch.nn.functional.conv2d(pred, weights_var)
    # grad = torch.sum(grad, dim=1, keepdim=True)
    sloss = charbonnier_diff(grad).mean()

    return sloss
