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

import time
import sys
from colorama import Fore


def timer(func, *args, repeats=1, **kwargs):
    start_time = time.time()
    for i in range(repeats):
        results = func(*args, **kwargs)
    #
    elapsed_time = (time.time() - start_time)/repeats
    return results, elapsed_time


# measures time spent on the current process.
# this is more accurate, but this will not measure time spent in os.system() calls
# so use timer() if you have os.system() calls.
def process_timer(func, *args, repeats=1, **kwargs):
    start_time = time.clock()
    for i in range(repeats):
        results = func(*args, **kwargs)
    #
    elapsed_process_time = (time.clock() - start_time)/repeats
    return results, elapsed_process_time


def delta_time(seconds):
    days, seconds = divmod(seconds,(60*60*24))
    hours, seconds = divmod(seconds,(60*60))
    minutes, seconds = divmod(seconds,60)
    return days, hours, minutes, seconds


def delta_time_string(seconds):
    days, hours, minutes, seconds = delta_time(seconds)
    time_str = f'{minutes:02.0f}:{seconds:02.0f}'
    time_str = f'{hours:02.0f}:{time_str}' if hours > 0 else time_str
    time_str = f'{days:1.0f}d,{time_str}' if days > 0 else time_str
    return time_str


def display_time_bar(desc, num_completed, total, start_time, end_time, file=None, colors=None):
    file = file if file is not None else sys.stdout
    time_taken_str = eta_str = it_per_sec = ''
    time_delta = end_time - start_time
    if num_completed > 0 and time_delta > 0 and total is not None:
        time_taken_str = delta_time_string(time_delta)
        eta_str = delta_time_string(time_delta*(total-num_completed)/num_completed)
        it_per_sec = f'{(time_delta/num_completed):5.2f}s/it' if (time_delta > num_completed) \
            else f'{(num_completed/time_delta):5.2f}it/s'
    #
    num_bars = int(num_completed*10.0/total) if total is not None else 0
    percentage = f'{num_completed*100.0/total:5.0f}%' if total is not None else '    '
    bar_string = f"{'#'*num_bars + ' '*(10-num_bars)}"
    if colors is not None:
        assert len(colors) == 4, f'colors must have length 4'
        file.write(f'\r{colors[0]}{desc}|'
                   f'{colors[1]}{bar_string}| '
                   f'{colors[2]}{percentage} {num_completed}/{total}| '
                   f'{colors[3]}[{time_taken_str}<{eta_str} {it_per_sec}]{Fore.RESET}')
    else:
        file.write(f'\r{desc}|'
                   f'{bar_string}| '
                   f'{percentage} {num_completed}/{total}| '
                   f'[{time_taken_str}<{eta_str} {it_per_sec}]')
    #
    file.flush()
