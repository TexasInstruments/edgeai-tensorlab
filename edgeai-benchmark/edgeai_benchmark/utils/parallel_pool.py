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

import os
import sys
import multiprocessing
from multiprocessing import pool
import collections
import time
import traceback
from .progress_step import *
from .logger_utils import *


class NoDaeomonPool(pool.Pool):
    # daemon processes are not allowed to have children
    # a hack to create non-daemon process in multiprocessing.Pool
    def Process(self, ctx, *args, **kwargs):
        class NoDaemonProcess(ctx.Process):
            def _get_daemon(self):
                return False
            def _set_daemon(self, value):
                pass
            daemon = property(_get_daemon, _set_daemon)
        return NoDaemonProcess(*args, **kwargs)


class ParallelPool:
    def __init__(self, num_processes, parallel_devices=None, desc='tasks', blocking=True, maxinterval=10.0):
        self.desc = desc
        self.num_processes = num_processes
        self.parallel_devices = parallel_devices
        self.queued_tasks = collections.deque()
        self.maxinterval = maxinterval
        self.blocking = blocking
        assert self.parallel_devices is None or len(self.parallel_devices) == num_processes, \
            f'length of parallel_devices {self.parallel_devices} must match num_processes {num_processes}'

    def enqueue(self, task):
        self.queued_tasks.append(task)

    def run(self):
        assert len(self.queued_tasks) > 0, f'at least one task must be queued, got {len(self.queued_tasks)}'
        return self._run_parallel()

    def _run_sequential(self):
        result_list = []
        for task_id, task in progress_step(self.queued_tasks, desc='tasks'):
            result = task()
            result_list.append(result)
        #
        return result_list

    def _run_parallel(self):
        PoolType = pool.Pool #NoDaeomonPool
        with PoolType(self.num_processes) as process_pool:
            results_iterator = process_pool.imap_unordered(self._worker, self.queued_tasks)
            result_list = []
            num_completed = num_completed_prev = 0
            num_tasks = len(self.queued_tasks)
            pbar_tasks = progress_step(iterable=range(num_tasks), desc=self.desc, position=1)
            while num_completed < num_tasks:
                # check if a result is available
                try:
                    result = results_iterator.__next__(timeout=self.maxinterval)
                    result_list.append(result)
                except multiprocessing.TimeoutError as e:
                    pass
                #
                num_completed = len(result_list)
                if num_completed > num_completed_prev:
                    pbar_tasks.update(num_completed-num_completed_prev)
                    num_completed_prev = num_completed
                #
            #
            pbar_tasks.close()
            print('\n')
            return result_list

    def _worker(self, task):
        if self.parallel_devices is not None:
            current_process = multiprocessing.current_process()
            process_index = current_process._identity[0] - 1
            # if a task crashes, the process will be re-created and will have a new id assigned
            # hence, the process_index can be higher than num_processes
            parallel_device = self.parallel_devices[process_index%self.num_processes]
            os.environ['CUDA_VISIBLE_DEVICES'] = str(parallel_device)
            print(log_color('\nINFO', 'starting process on parallel_device', parallel_device))
        #
        return task()


class MultiProcessingTaskMaker:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args_init = args
        self.kwargs_init = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*self.args_init, **self.kwargs_init)


def process_run(func, *args, **kwargs):
    with NoDaeomonPool(1) as process_pool:
        task = MultiProcessingTaskMaker(func, *args, **kwargs)
        results = process_pool.map(task, [0])
        return results[0]
