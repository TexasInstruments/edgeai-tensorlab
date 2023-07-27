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
import queue
import copy

from .progress_step import *
from .logger_utils import *


class ParallelRun:
    def __init__(self, num_processes, parallel_devices=None, desc='tasks', blocking=True, maxinterval=0.1):
        self.desc = desc
        self.num_processes = num_processes
        self.parallel_devices = parallel_devices
        self.queued_tasks = collections.deque()
        self.maxinterval = maxinterval
        self.blocking = blocking

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
        result_list = []
        num_total_tasks = len(self.queued_tasks)
        pbar_tasks = progress_step(iterable=range(num_total_tasks), desc=self.desc, position=1)
        while len(self.queued_tasks) > 0:
            try:
                result_list1 = self._run_parallel_loop(pbar_tasks)
            except Exception as exception_e:
                print(f"Exception occurred in parllel loop: {exception_e} \nRestarting the parallel loop - remaining tasks: {len(self.queued_tasks)}")
                traceback.print_exc()
            else:
                result_list.append(result_list1)
            #
        #
        pbar_tasks.close()
        print('\n')
        return result_list

    def _run_parallel_loop(self, pbar_tasks):
        result_list = []
        num_queued_tasks = len(self.queued_tasks)
        num_started_tasks = 0

        result_queues_dict = dict()
        process_dict = dict()
        mp_context = multiprocessing.get_context(method="spawn") #fork, forkserver, spawn
        while len(result_list) < num_queued_tasks:
            if len(process_dict) < self.num_processes and len(self.queued_tasks) > 0:
                task = self.queued_tasks.pop()
                result_queue = mp_context.Queue()
                proc = mp_context.Process(target=self._worker, args=(task,num_started_tasks,result_queue))
                proc.start()
                result_queues_dict[num_started_tasks] = result_queue
                process_dict[num_started_tasks] = proc
                num_started_tasks += 1
            #

            result_queues_dict_shallow_copy = copy.copy(result_queues_dict)
            for r_key, r_queue in result_queues_dict_shallow_copy.items():
                result = {}
                try:
                    (result, exception_e) = r_queue.get_nowait()
                except queue.Empty:
                    if not process_dict[r_key].is_alive():
                        process_dict.pop(r_key)
                        result_queues_dict.pop(r_key)
                        result_list.append({})
                        proc.terminate()
                        proc.join()
                    #
                    continue
                else:
                    if isinstance(exception_e, Exception):
                        #print(f"Exception occurred in child process: {exception_e}")
                        pass
                    #
                    result_list.append(result)
                    result_queues_dict.pop(r_key)
                    proc = process_dict.pop(r_key)
                    pbar_tasks.update(1)
                    proc.join()
                #
            #
            time.sleep(self.maxinterval)
        #
        return result_list

    def _worker(self, task, task_index, result_queue):
        result = {}
        exception_e = None
        try:
            if self.parallel_devices is not None:
                num_devices = len(self.parallel_devices)
                parallel_device = self.parallel_devices[task_index%num_devices]
                os.environ['CUDA_VISIBLE_DEVICES'] = str(parallel_device)
                print(log_color('\nINFO', 'starting process on parallel_device', parallel_device))
            #
            result = task()
        except Exception as e:
            print(f"Exception occurred in worker process: {e}")
            traceback.print_exc()
            exception_e = e
        #
        result_queue.put((result,exception_e))


