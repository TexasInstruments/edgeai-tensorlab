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

import random
import torch
from ..layers import functional


########################################################################
# pytorch implementation of a single tensor range
########################################################################

def extrema_fast(src, range_shrink_percentile=0.0, channel_mean=False, sigma=0.0, fast_mode=True):
    return extrema(src, range_shrink_percentile, channel_mean, sigma, fast_mode)


def extrema(src, range_shrink_percentile=0.0, channel_mean=False, sigma=0.0, fast_mode=False):
    if range_shrink_percentile == 0 and sigma == 0 and channel_mean == False:
        mn = src.min()
        mx = src.max()
        return mn, mx
    elif range_shrink_percentile:
        # downsample for fast_mode
        hist_array, mn, mx, mult_factor, offset = tensor_histogram(src, fast_mode=fast_mode)
        if hist_array is None:
            return mn, mx

        new_mn_scaled, new_mx_scaled = extrema_hist_search(hist_array, range_shrink_percentile)
        new_mn = (new_mn_scaled / mult_factor) + offset
        new_mx = (new_mx_scaled / mult_factor) + offset

        # take care of floating point inaccuracies that can
        # increase the range (in rare cases) beyond the actual range.
        new_mn = max(mn, new_mn)
        new_mx = min(mx, new_mx)
        return new_mn, new_mx
    elif channel_mean:
        dim = [0,2,3] if src.dim() == 4 else None
        mn = torch.amin(src, dim=dim, keepdim=False).mean()
        mx = torch.amax(src, dim=dim, keepdim=False).mean()
        return mn, mx
    elif sigma:
        mean = torch.mean(src)
        std = torch.std(src)
        mn = mean - sigma*std
        mx = mean + sigma*std
        return mn, mx
    else:
        assert False, 'unknown extrema computation mode'


def tensor_histogram(src, fast_mode=False):
    # downsample for fast_mode
    fast_stride = 2
    fast_stride2 = fast_stride * 2
    if fast_mode and len(src.size()) == 4 and (src.size(2) > fast_stride2) and (src.size(3) > fast_stride2):
        r_start = random.randint(0, fast_stride - 1)
        c_start = random.randint(0, fast_stride - 1)
        src = src[..., r_start::fast_stride, c_start::fast_stride]
    #
    mn = src.min()
    mx = src.max()
    if mn == 0 and mx == 0:
        return None, mn, mx, 1.0, 0.0
    #

    # compute range_shrink_percentile based min/max
    # frequency - bincount can only operate on unsigned
    num_bins = 255.0
    cum_freq = float(100.0)
    offset = mn
    range_val = torch.abs(mx - mn)
    mult_factor = (num_bins / range_val)
    tensor_int = (src.contiguous().view(-1) - offset) * mult_factor
    tensor_int = functional.round_g(tensor_int).int()

    # numpy version
    # hist = np.bincount(tensor_int.cpu().numpy())
    # hist_sum = np.sum(hist)
    # hist_array = hist.astype(np.float32) * cum_freq / float(hist_sum)

    # torch version
    hist = torch.bincount(tensor_int)
    hist_sum = torch.sum(hist)
    hist = hist.float() * cum_freq / hist_sum.float()
    hist_array = hist.cpu().numpy()
    return hist_array, mn, mx, mult_factor, offset


# this code is not parallelizable. better to pass a numpy array
def extrema_hist_search(hist_array, range_shrink_percentile):
    new_mn_scaled = 0
    new_mx_scaled = len(hist_array) - 1
    hist_sum_left = 0.0
    hist_sum_right = 0.0
    for h_idx in range(len(hist_array)):
        r_idx = len(hist_array) - 1 - h_idx
        hist_sum_left += hist_array[h_idx]
        hist_sum_right += hist_array[r_idx]
        if hist_sum_left < range_shrink_percentile:
            new_mn_scaled = h_idx
        if hist_sum_right < range_shrink_percentile:
            new_mx_scaled = r_idx
        #
    #
    return new_mn_scaled, new_mx_scaled

