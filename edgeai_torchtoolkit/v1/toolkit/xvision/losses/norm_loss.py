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
from .basic_loss import *
from .loss_utils import *

########################################
class L1NormDiff(BasicNormLossModule):
    def __init__(self, sparse=False, error_fn=l1_norm, error_name='L1NormDiff'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name)
supervised_l1_loss = l1_norm_loss = L1NormDiff


class L2NormDiff(BasicNormLossModule):
    def __init__(self, sparse=False, error_fn=l2_norm, error_name='L2NormDiff'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name)
supervised_l2_loss = l2_norm_loss = L2NormDiff


#take absolute norm of tensor as loss instead of diff
class L2NormSelf(BasicNormLossModule):
    def __init__(self, sparse=False, error_fn=l2_norm_self, error_name='L2NormSelf'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name)
supervised_l2_loss_self = l2_norm_loss_self = L2NormSelf


########################################
class SmoothL1Diff(BasicElementwiseLossModule):
    def __init__(self, sparse=False, error_fn=smooth_l1_loss, error_name='SmoothL1Diff'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name)
supervised_error = supervised_loss = supervised_smooth_l1_loss = smooth_l1_norm_loss = SmoothL1Diff


class CharbonnierDiff(BasicElementwiseLossModule):
    def __init__(self, sparse=False, error_fn=charbonnier, error_name='CharbonnierDiff'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name)
charbonnier_loss = SupervisedLoss = CharbonnierDiff


class CharbonnierAdaptiveDiff(BasicElementwiseLossModule):
    def __init__(self, sparse=False, error_fn=charbonnier_adaptive, error_name='CharbonnierAdaptiveDiff'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name)
charbonnier_adaptive_loss = SupervisedAdaptiveLoss = CharbonnierAdaptiveDiff


class BerHuDiff(BasicElementwiseLossModule):
    def __init__(self, sparse=False, error_fn=berhu, error_name='BerHuDiff'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name)

########################################
class RootMeanSquaredError(BasicElementwiseLossModule):
    def __init__(self, sparse=False, error_fn=square_diff, error_name='RMSE'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name, square_root=True)
supervised_root_mean_squared_error = root_mean_squared_error = RootMeanSquaredError

########################################
class AbsRelDiff(BasicElementwiseLossModule):
    def __init__(self, sparse=False, error_fn=abs_relative_diff, error_name='ARDiff'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name)
supervised_relative_error = abs_relative_error = AbsRelDiff


class AbsRelDiffX100(BasicElementwiseLossModule):
    def __init__(self, sparse=False, error_fn=abs_relative_diff_x100, error_name='AbsRelDiffX100'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name)
supervised_relative_error_x100 = abs_relative_error_x100 = AbsRelDiffX100


def abs_relative_diff_rng1(x, y):
    return abs_relative_diff(x, y, eps = 1.0)

class AbsRelDiffRng1(BasicElementwiseLossModule):
    def __init__(self, sparse=False, error_fn=abs_relative_diff_rng1, error_name='ARDiffMin1'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name, min_elements=2)
supervised_relative_error_rng1 = abs_relative_error_rng1 = AbsRelDiffRng1


def abs_relative_diff_rng3to20(x, y):
    return abs_relative_diff(x, y, eps = 3.0, max_val=20)

class AbsRelDiffRng3To20(BasicElementwiseLossModule):
    def __init__(self, sparse=False, error_fn=abs_relative_diff_rng3to20, error_name='ARDiff3To20'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name, min_elements=2)
supervised_relative_error_rng3to20 = abs_relative_error_rng3to20 = AbsRelDiffRng3To20


def abs_relative_diff_rng3to80(x, y):
    return abs_relative_diff(x, y, eps = 3.0, max_val=80)

class AbsRelDiffRng3To80(BasicElementwiseLossModule):
    def __init__(self, sparse=False, error_fn=abs_relative_diff_rng3to80, error_name='ARDiff3To80'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name, min_elements=2)
supervised_relative_error_rng3to80 = abs_relative_error_rng3to80 = AbsRelDiffRng3To80


def abs_relative_diff_rng3to255(x, y):
    return abs_relative_diff(x, y, eps = 3.0, max_val=255)

def abs_relative_diff_weighted_rng3to255(x, y):
    return abs_relative_diff_weighted(x, y, eps = 3.0, max_val=255, weights=[1.0, 0.1])

class AbsRelDiffRng3To255(BasicElementwiseLossModule):
    def __init__(self, sparse=False, error_fn=abs_relative_diff_rng3to255, error_name='ARDiff3To255'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name, min_elements=2)
supervised_relative_error_rng3to255 = abs_relative_error_rng3to255 = AbsRelDiffRng3To255

class AbsRelDiffWeightedRng3To255(BasicElementwiseLossModule):
    def __init__(self, sparse=False, error_fn=abs_relative_diff_weighted_rng3to255, error_name='ARDiffWeighted3To255'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name, min_elements=2)
supervised_relative_error_weighted_rng3to255 = abs_relative_error_weighted_rng3to255 = AbsRelDiffWeightedRng3To255

########################################
class ErrorVar(BasicElementwiseLossModule):
    def __init__(self, sparse=False, error_fn=error_variance, error_name='ErrorVar'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name)
supervised_error_var = ErrorVar

class ErrorSTD(BasicElementwiseLossModule):
    def __init__(self, sparse=False, error_fn=error_std, error_name='ErrorSTD'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name)
supervised_error_std = ErrorSTD