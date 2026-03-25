# Copyright (c) 2018-2025, Texas Instruments
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

import os
import sys
import multiprocessing
import collections
import time
import traceback
import queue
import copy
import functools

from .progress_step import *
from .logger_utils import *


# Create a multiprocessing context with spawn method
mp_context = multiprocessing.get_context('spawn')


class ProcessWithQueue(mp_context.Process):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None,
        result_queue=None, log_file=None, **proc_kwargs):
        if kwargs is None:
            kwargs = {}
        if result_queue is None:
            result_queue = mp_context.Queue()
            
        kwargs = copy.copy(kwargs)
        target = functools.partial(self._worker, target, result_queue)
        super().__init__(group, target, name, args, kwargs, **proc_kwargs)
        self.log_file = log_file
        self.returncode = None
        self.result_queue = result_queue

    def wait(self, input=None, timeout=None):
        try:
            self.join(timeout=timeout)
        except mp_context.TimeoutError:
            if self.result_queue.empty():
                raise mp_context.TimeoutError
            #
        except:
            raise RuntimeError(f'Error during wait() in {__file__}')
        #
        
        # join() doesn't seem to be raising TimeoutError, so check the exitcode
        # when timeout occurs in join(), exitcode will have None
        if self.exitcode is None:
            raise mp_context.TimeoutError
        #
        self.returncode = self.exitcode
        return self.exitcode

    def communicate(self, input=None, timeout=None):
        try:
            self.wait(timeout=timeout)
        except mp_context.TimeoutError:
            if self.result_queue.empty():
                raise mp_context.TimeoutError
            #
        except:
            raise RuntimeError(f'Error during communicate() in {__file__}')
        #
        result, exception_e = self.result_queue.get()
        self.join()
        return result, exception_e

    def _worker(self, task, result_queue, **kwargs):
        result = {}
        exception_e = None
        try:
            result = task(**kwargs)
        except KeyboardInterrupt:
            print(f"KeyboardInterrupt occurred in worker process: {__file__}")
            traceback.print_exc()
            exception_e = None
            raise
        except Exception as e:
            print(f"Exception occurred in worker process: {e}")
            traceback.print_exc()
            exception_e = e
        #
        result_queue.put((result,exception_e))
