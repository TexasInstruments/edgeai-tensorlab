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
import time
import traceback
import subprocess
import tqdm
import warnings


class ParallelSubProcess:
    def __init__(self, parallel_processes, parallel_devices=None, desc='TASKS', maxinterval=1.0, tqdm_obj=None, verbose=False):
        self.parallel_processes = parallel_processes
        self.parallel_devices = parallel_devices if isinstance(parallel_devices, (list,tuple)) else list(range(parallel_devices))
        self.desc = desc
        self.queued_tasks = dict()
        self.proc_dict = dict()
        self.tqdm_obj = tqdm_obj
        self.maxinterval = maxinterval
        self.num_queued_tasks = 0
        self.task_index = 0
        if verbose:
            warnings.warn('''
            ParallelSubProcess is for tasks that are started with subprocess.Popen()
            The task should return the proc object that supports communicate
            For more details, see: https://docs.python.org/3/library/subprocess.html
            ''')
        #

    def enqueue(self, task_name, task_list):
        if not isinstance(task_list, (list,tuple)):
            task_list = [task_list]
        #
        self.queued_tasks.update({task_name:task_list})
        self.num_queued_tasks += 1

    def run(self):
        desc = self.desc + f' STATUS - TOTAL={self.num_queued_tasks}, NUM_RUNNING={0}'
        if self.tqdm_obj is None:
            self.tqdm_obj = tqdm.tqdm(total=self.num_queued_tasks, position=0, desc=desc)
        #
        num_completed, num_running = self._wait_in_loop(self.parallel_processes)
        while num_completed < self.num_queued_tasks:
            # wait in a loop until the number of running processes comes down
            num_completed, num_running = self._wait_in_loop(self.parallel_processes)
            proc_dict_to_start = self._find_proc_dict_to_start()
            # start the process
            if proc_dict_to_start:
                proc_dict_to_start['proc'] = self._worker(proc_dict_to_start['proc_func'])
                proc_dict_to_start['running'] = True
            #
        #
        # wati for all remaining processes to finish
        self._wait_in_loop(0)
        return True

    def _find_proc_dict_to_start(self):
        proc_dict_to_start = None
        # now search through tasks and find a process to start
        for task_name, task_list in self.queued_tasks.items():
            completed_flag_in_task = []
            for proc_id, proc_dict in enumerate(task_list):
                if len(completed_flag_in_task) == 0 or all(completed_flag_in_task):
                    # if all the proc in this task untill now are complete
                    # we can go and check the next proc in this task
                    running = proc_dict.get('running', False)
                    completed = proc_dict.get('completed', False)
                    completed_flag_in_task.append(completed)
                    if (not running) and (not completed):
                        proc_dict_to_start = proc_dict
                        break
                    #
                #
            #
            if proc_dict_to_start:
                break
            #
        #
        return proc_dict_to_start

    def _check_proc_complete(self, proc):
        completed = False
        exit_code = proc.returncode
        try:
            out_ret, err_ret = proc.communicate(timeout=0.1)
        except subprocess.TimeoutExpired as ex:
            pass
        else:
            completed = True
        #
        return completed

    def _check_running_status(self):
        running_tasks = []
        completed_tasks = []
        for task_name, task_list in self.queued_tasks.items():
            completed_proc_in_task = []
            running_proc_in_task = []
            running_proc_name = None
            for proc_id, proc_dict in enumerate(task_list):
                proc = proc_dict.get('proc', None)
                completed = (proc is not None) and proc_dict.get('completed', False)
                running = (proc is not None) and proc_dict.get('running', False)
                if running:
                    # try to update the completed status for running processes
                    completed = self._check_proc_complete(proc)
                    running = (not completed)
                    proc_dict['completed'] = completed
                    proc_dict['running'] = running
                    running_proc_name = proc_dict['proc_name']
                #
                completed_proc_in_task.append(completed)
                running_proc_in_task.append(running)
            #
            is_task_completed = all(completed_proc_in_task)
            is_task_running = any(running_proc_in_task)
            if is_task_completed:
                completed_tasks += [task_name]
            #
            if is_task_running:
                running_tasks += [running_proc_name or task_name]
            #
        #

        num_completed = len(completed_tasks)
        num_running = len(running_tasks)

        self.tqdm_obj.update(num_completed - self.tqdm_obj.n)
        desc = self.desc + f' STATUS - TOTAL={self.num_queued_tasks}, NUM_RUNNING={num_running}'
        self.tqdm_obj.set_description(desc)
        self.tqdm_obj.set_postfix(postfix=dict(RUNNING=running_tasks, COMPLETED=completed_tasks))
        return num_completed, num_running

    def _wait_in_loop(self, num_processes):
        # wait in a loop until the number of running processes come down
        num_completed, num_running = self._check_running_status()
        while num_running > 0 and num_running >= num_processes:
            num_completed, num_running = self._check_running_status()
            time.sleep(self.maxinterval)
        #
        return num_completed, num_running

    def _worker(self, task):
        proc = None
        try:
            if self.parallel_devices is not None:
                num_devices = len(self.parallel_devices)
                parallel_device = self.parallel_devices[self.task_index%num_devices]
                os.environ['CUDA_VISIBLE_DEVICES'] = str(parallel_device)
            #
            proc = task()
        except Exception as e:
            print(f"Exception occurred in worker process: {e}")
            traceback.print_exc()
        #
        self.task_index += 1
        return proc
