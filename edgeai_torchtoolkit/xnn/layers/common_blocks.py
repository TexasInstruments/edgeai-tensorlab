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

import torch
from . import functional

###########################################################
# add
class AddBlock(torch.nn.Module):
    def __init__(self, inplace=False, signed=True, *args, **kwargs):
        super().__init__()
        self.inplace = inplace
        self.signed = signed

    def forward(self, x):
        assert isinstance(x, (list,tuple)), 'input to add block must be a list or tuple'
        y = x[0]
        for i in range(1,len(x)):
            y = y + x[i]
        #
        return y

    def __repr__(self):
        return 'AddBlock(inplace={}, signed={})'.format(self.inplace, self.signed)

# sub
class SubtractBlock(torch.nn.Module):
    def __init__(self, inplace=False, signed=True, *args, **kwargs):
        super().__init__()
        self.inplace = inplace
        self.signed = signed

    def forward(self, x):
        assert isinstance(x, (list,tuple)), 'input to sub block must be a list or tuple'
        y = x[0]
        for i in range(1,len(x)):
            y = y - x[i]
        #
        return y

    def __repr__(self):
        return 'SubtractBlock(inplace={}, signed={})'.format(self.inplace, self.signed)


###########################################################
# mult
class MultBlock(torch.nn.Module):
    def __init__(self, inplace=False, signed=True, *args, **kwargs):
        super().__init__()
        self.inplace = inplace
        self.signed = signed

    def forward(self, x):
        assert isinstance(x, (list,tuple)), 'input to add block must be a list or tuple'
        y = x[0]
        for i in range(1,len(x)):
            y = y * x[i]
        #
        return y

    def __repr__(self):
        return 'MultBlock(inplace={}, signed={})'.format(self.inplace, self.signed)


###########################################################
# cat
class CatBlock(torch.nn.Module):
    def __init__(self, inplace=False, signed=True, dim=1, *args, **kwargs):
        super().__init__()
        self.inplace = inplace
        self.signed = signed
        self.dim = dim

    def forward(self, x):
        assert isinstance(x, (list,tuple)), 'input to add block must be a list or tuple'
        y = torch.cat(x, dim=self.dim)
        return y

    def __repr__(self):
        return 'CatBlock(inplace={}, signed={})'.format(self.inplace, self.signed)


###########################################################
# moving sum
class MovingSumBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prev_x = 0

    def forward(self, x):
        y = x + self.prev_x
        self.prev_x = x
        return y


###########################################################
# a bypass block that does nothing - can be used as placeholder
BypassBlock = torch.nn.Identity



###########################################################
# convert to linear view that van be given to a fully connected layer
ViewAsLinear = torch.nn.Flatten


###########################################################
# Split channel-wise and return the first part
class SplitChannelsTakeFirst(torch.nn.Module):
    def __init__(self, splits=2):
        super().__init__()
        self.splits = splits

    def forward(self, x):
        parts = functional.channel_split_by_chunks(x, self.splits)
        return parts[0]

    def __repr__(self):
        return 'SplitChannelsTakeFirst(splits={})'.format(self.splits)


###########################################################
# Split channel-wise and return the first part
class SplitChannelsTakeLast(torch.nn.Module):
    def __init__(self, splits=2):
        super().__init__()
        self.splits = splits

    def forward(self, x):
        parts = functional.channel_split_by_chunks(x, self.splits)
        return parts[-1]

    def __repr__(self):
        return 'SplitChannelsTakeLast(splits={})'.format(self.splits)


###########################################################
# Split channel-wise and add the parts
class SplitChannelsAdd(torch.nn.Module):
    def __init__(self, splits=2):
        super().__init__()
        self.splits = splits

    def forward(self, x):
        parts = functional.channel_split_by_chunks(x, self.splits)
        sum = parts[0]
        for part in parts[1:]:
            sum = sum + part
        #
        return sum

    def __repr__(self):
        return 'SplitChannelsAdd(splits={})'.format(self.splits)


###########################################################
# Split channel-wise and return the first part
class SplitListTakeFirst(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0]

    def __repr__(self):
        return 'SplitListTakeFirst()'


###############################################################
# Parallel as oposed to Sequential
class ParallelBlock(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        assert (len(args)==2), 'for now supporting only two modules in parallel'
        #store it as modulelist for cuda() to work
        self.blocks = torch.nn.ModuleList(args)
        self.split = len(self.blocks)

    def forward(self, x):
        x_spits = functional.channel_split_by_chunks(x, self.split)
        out_splits = [None]*self.split
        for id, blk in enumerate(self.blocks):
            out_splits[id] = blk(x_spits[id])

        x = torch.cat(out_splits, dim=1)
        return x



###############################################################
class ShuffleBlock(torch.nn.Module):
    def __init__(self,groups):
        super().__init__()
        self.groups = groups
    def forward(self,x):
         if self.groups > 1:
             return functional.channel_shuffle(x, groups=self.groups)
         else:
             return x


###############################################################
class ArgMax(torch.nn.Module):
    def __init__(self, dim=1, keepdim=True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
    def forward(self, x):
        y = torch.argmax(x, dim=self.dim, keepdim=self.keepdim)
        return y