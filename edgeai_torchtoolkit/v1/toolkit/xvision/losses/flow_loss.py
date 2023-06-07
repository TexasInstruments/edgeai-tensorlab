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


__all__ = [
    'end_point_error', 'end_point_loss', 'outlier_fraction', 'outlier_precentage'
]



############################################################################
class EPError(BasicNormLossModule):
    def __init__(self, sparse=False, error_fn=l2_norm, error_name='EPError'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name)


end_point_error = EPError
end_point_loss = EPError


############################################################################
def outlier_check(prediction, target, absolute_thr=3.0, relative_thr=0.05):
    norm_dist = l2_norm(prediction, target)
    norm_pred = l2_norm(prediction)
    norm_target = l2_norm(target)

    eps_arr = (norm_target == 0).float() * (1e-6)   # To avoid division by zero.
    rel_dist = norm_pred / (norm_target + eps_arr)

    is_outlier = ((norm_dist > absolute_thr) & (rel_dist > relative_thr)).float()
    return is_outlier


def outlier_check_x100(opt, target, absolute_thr=3.0, relative_thr=0.05):
    return outlier_check(opt, target, absolute_thr, relative_thr) * 100.0


class OutlierFraction(BasicNormLossModule):
    def __init__(self, sparse=False, error_fn=outlier_check, error_name='OutlierFraction'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name)
#
outlier_fraction = OutlierFraction


class OutlierPercentage(BasicNormLossModule):
    def __init__(self, sparse=False, error_fn=outlier_check_x100, error_name='OutlierPercentage'):
        super().__init__(sparse=sparse, error_fn=error_fn, error_name=error_name)
#
outlier_precentage = OutlierPercentage


############################################################################
