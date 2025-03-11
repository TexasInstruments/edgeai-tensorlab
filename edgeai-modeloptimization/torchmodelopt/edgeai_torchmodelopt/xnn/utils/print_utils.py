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

from colorama import Fore
import numpy as np
import copy

__all__ = ['AverageMeter', 'add_color', 'print_once', 'print_color', 'print_yellow', 'print_red', 'print_green', 'print_blue']


##################################################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, print_avg=True):
        self.print_avg = print_avg
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += (val * n) if not np.isnan(val) else 0
        self.count += (n) if not np.isnan(val) else 0
        self.avg = (self.sum / self.count) if self.count!=0 else 0

    def get(self):
        return (self.avg if self.print_avg else self.val)

    def __float__(self):
        return self.get()

    def __str__(self):
        output = '{:.4g}'.format(self.get())
        return output

    __repr__ = __str__

    def string(self):
        output = 'Avg:{:.3g},Val:{:.3g}'.format(self.avg, self.val)
        return output


##################################################################################
def add_color(string, color=None):
    if color:
        string = '{}{}{}'.format(color, string, Fore.RESET)
    #
    return string
#

def print_color(string, *args, **kwargs):
    if 'color' in kwargs:
        string = add_color(string, kwargs['color'])
        kwargs_copy = copy.deepcopy(kwargs)
        del kwargs_copy['color']
    else:
        kwargs_copy = kwargs
    #
    print(string, *args, **kwargs_copy)
#


print_once_dict = {}
def print_once(string, *args, **kwargs):
    global print_once_dict
    if string not in list(print_once_dict.keys()):
        print_color(string, *args, **kwargs)
        print_once_dict[string] = True
    #
    return


def print_yellow_once(string, *args, **kwargs):
    print_once(string, *args, **kwargs, color=Fore.YELLOW)


def print_yellow(string, *args, **kwargs):
    print_color(string, *args, **kwargs, color=Fore.YELLOW)


def print_red(string, *args, **kwargs):
    print_color(string, *args, **kwargs, color=Fore.RED)


def print_green(string, *args, **kwargs):
    print_color(string, *args, **kwargs, color=Fore.GREEN)


def print_blue(string, *args, **kwargs):
    print_color(string, *args, **kwargs, color=Fore.BLUE)


