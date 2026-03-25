# Copyright (c) 2018-2021, Texas Instruments
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

import sys
import time
from colorama import Fore
from tqdm import tqdm
from .timer_utils import display_time_bar

__all__ = ['progress_step', 'progress_step_tqdm', 'progress_step_lite']


######################################################################
# utility functions
def _progress_miniters(iterable, total=None):
    iter_length = len(iterable) if hasattr(iterable, '__len__') else total
    miniters = max(iter_length//100, 1) if iter_length is not None else 1
    return miniters


def _progress_format(desc_len=60, color_map=None):
    if color_map is not None:
        assert len(color_map) == 4, f'color_map must have length 4'
        format_arg = (color_map[0], desc_len, color_map[1], color_map[2], color_map[3], Fore.RESET)
        bar_format = '%s{desc:%s}|%s{percentage:4.0f}%%|%s{bar:10}|%s{r_bar}%s' % format_arg
    else:
        bar_format = '{desc:%s}|{percentage:4.0f}%%|{bar:10}|{r_bar}' % desc_len
    #
    return bar_format


def _progress_desc(desc, desc_len=60):
    desc = desc[:desc_len] if (desc_len is not None and len(desc) > desc_len) else \
        desc + ' '*(desc_len-len(desc))
    return desc


######################################################################
def progress_step_tqdm(iterable, desc, total=None, miniters=None, mininterval=10.0, bar_format=None,
                       desc_len=60, file=sys.stdout, leave=True, color_map=None, **kwargs):
    """
    Uses a tqdm variant that updates only once in a while
    """
    return TqdmStep(iterable=iterable, desc=desc, total=total, miniters=miniters, mininterval=mininterval,
                bar_format=bar_format, file=file, leave=leave, desc_len=desc_len, color_map=color_map, **kwargs)


class TqdmStep(tqdm):
    """
    A tqdm variant that updates even before the first iteration,
    and also updates only once in a while
    """
    def __init__(self, iterable, desc=None, total=None, bar_format=None, miniters=None, mininterval=30.0,
                 desc_len=60, color_map=None, **kwargs):
        desc = _progress_desc(desc, desc_len)
        # somehow controlling update interval with miniters doesn't work (gets overwritten to 1)
        # miniters = _progress_miniters(iterable, total) if miniters is None else miniters
        bar_format = _progress_format(desc_len, color_map) if bar_format is None else bar_format
        super().__init__(iterable, desc=desc, total=total, bar_format=bar_format, miniters=miniters, mininterval=mininterval, **kwargs)
        # display bar even before the first iteration. useful if the first iteration itself  takes some time
        display_time_bar(desc, num_completed=0, total=self.total, start_time=0, end_time=0, file=self.fp)


######################################################################
# a lighter version of progress_step that doesn't use tqdm
# this prints the iteration descriptor even before the first iteration
def progress_step_lite(iterable, desc, desc_len=60, total=None, miniters=None, color_map=None, **kwargs):
    return ProgressStepLite(iterable, desc, total=total, miniters=miniters, color_map=color_map, desc_len=desc_len, **kwargs)


class ProgressStepLite:
    """
    A simple progress indicator that can be used instead of tqdm
    This has an advantage that this starts updating the status in the 0th iteration
    (i.e. before even first iteration is complete)
    Author: Manu Mathew
    2021 Feb 16
    """
    def __init__(self, iterable, desc, total=None, miniters=1, file=None,
                 desc_len=60, color_map=None, position=0, **kwargs):
        super().__init__()
        self.total = iterable.__len__() if hasattr(iterable, '__len__') else total
        miniters = _progress_miniters(iterable, total) if miniters is None else miniters
        self.iterable = iterable
        self.desc = _progress_desc(desc, desc_len)
        self.step_size = miniters
        self.file = file if file is not None else sys.stdout
        self.color_map = color_map
        self.position = position
        self.num_completed = 0
        self.start_time = time.time()
        self.move_up_str = '\033[F' #'\x1b[A'
        self.move_down_str = '\n'

    def __iter__(self):
        self.start_time = time.time()
        self.update(0)
        for item_id, item in enumerate(self.iterable):
            yield item
            # when update is called from within the loop, it will display the bar
            # only at the specified step_size
            self.update(1, force=False)
        #

    # when update is explicitly called, it will display the bar
    def update(self, n, force=True):
        self.num_completed += n
        end_time = time.time()
        if force or (self.num_completed % self.step_size) == 0 or (self.num_completed == self.total):
            self._set_position()
            display_time_bar(self.desc, self.num_completed, total=self.total, start_time=self.start_time,
                             end_time=end_time, file=self.file, color_map=self.color_map)
            self._reset_position()
        #

    def close(self):
        pass

    def _set_position(self):
        pos_str = self.move_up_str*self.position
        self.file.write(pos_str)
        self.file.flush()

    def _reset_position(self):
        pos_str = self.move_down_str*self.position
        self.file.write(pos_str)
        self.file.flush()


######################################################################
# the default progress_step
progress_step = progress_step_tqdm
