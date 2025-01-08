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


class ParallelProcess:
    def __init__(self, parallel_processes, parallel_devices=None, desc='tasks', blocking=True, verbose=True, maxinterval=60):
        self.desc = desc
        self.parallel_processes = parallel_processes
        self.parallel_devices = parallel_devices
        self.queued_tasks = collections.deque()
        self.maxinterval = maxinterval
        self.blocking = blocking
        self.verbose = verbose
        self.num_total_tasks = 0
        self.num_started_tasks = 0
        self.result_queues_dict = dict()
        self.process_dict = dict()
        self.result_list = []
        if self.verbose:
            print(log_color('\nINFO', "parallel_run", f"parallel_processes:{self.parallel_processes} parallel_devices={self.parallel_devices}"))
            sys.stdout.flush()
        #

    def enqueue(self, task):
        self.queued_tasks.append(task)

    def run(self):
        assert len(self.queued_tasks) > 0, f'at least one task must be queued, got {len(self.queued_tasks)}'
        return self._run_parallel()

    def _run_sequential(self):
        self.result_list = []
        for task_id, task in progress_step(self.queued_tasks, desc='tasks'):
            result = task()
            self.result_list.append(result)
        #
        return self.result_list

    def _run_parallel(self):
        self.result_list = []
        self.num_total_tasks = len(self.queued_tasks)
        self.num_started_tasks = 0
        self.result_queues_dict = dict()
        self.process_dict = dict()
        pbar_tasks = progress_step(iterable=range(self.num_total_tasks), desc=self.desc, position=1)
        while len(self.result_list) < self.num_total_tasks:
            try:
                self._run_parallel_loop(pbar_tasks)
            except Exception as exception_e:
                print(f"Exception occurred in parallel loop: {exception_e} \nRestarting the parallel loop - remaining tasks: {len(self.queued_tasks)}")
                traceback.print_exc()
                # add a dummy result because a process/task has exited unexpectedly
                self.result_list.append({})
                pbar_tasks.update(1)
            #
        #
        pbar_tasks.close()
        print('\n')
        return self.result_list

    def _run_parallel_loop(self, pbar_tasks):
        mp_context = multiprocessing.get_context(method="fork") #fork, forkserver, spawn
        last_time = time.time()

        while len(self.result_list) < self.num_total_tasks:
            cur_time = time.time()
            if self.verbose and (cur_time - last_time) >= self.maxinterval:
                print(log_color('\nINFO', "parallel_run", f"num_total_tasks:{self.num_total_tasks} "
                      f"len(queued_tasks):{len(self.queued_tasks)} len(process_dict):{len(self.process_dict)} "
                      f"len(result_list):{len(self.result_list)}"))
                last_time = cur_time
            #

            # start the processes
            if len(self.process_dict) < self.parallel_processes and len(self.queued_tasks) > 0:
                task = self.queued_tasks.pop()
                result_queue = mp_context.SimpleQueue()
                proc = mp_context.Process(target=self._worker, args=(task,self.num_started_tasks,result_queue))
                proc.start()
                self.result_queues_dict[self.num_started_tasks] = result_queue
                self.process_dict[self.num_started_tasks] = proc
                self.num_started_tasks += 1
            #

            # collect the available the results
            result_queues_dict_shallow_copy = copy.copy(self.result_queues_dict)
            for r_key, r_queue in result_queues_dict_shallow_copy.items():
                result = {}
                exception_e = None
                if not r_queue.empty():
                    (result, exception_e) = r_queue.get()
                    self.result_list.append(result)
                    self.result_queues_dict.pop(r_key)
                    proc = self.process_dict.pop(r_key)
                    proc.join()
                    pbar_tasks.update(1)
                elif not self.process_dict[r_key].is_alive():
                    self.result_list.append(result)
                    self.result_queues_dict.pop(r_key)
                    proc = self.process_dict.pop(r_key)
                    proc.terminate() # something has happened with the process, terminate it.
                    proc.join()
                    pbar_tasks.update(1)
                #
            #

            time.sleep(0.1)
        #
        return self.result_list

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


