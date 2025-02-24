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
import multiprocessing
import tqdm
import warnings
import re
import wurlitzer


class SequentialRunner:
    def __init__(self, parallel_processes=None, desc='TASKS', mininterval=1.0, maxinterval=60.0, tqdm_obj=None,
            overall_timeout=None, instance_timeout=None, verbose=False):
        self.parallel_processes = parallel_processes
        self.desc = desc
        self.queued_tasks = dict()
        self.proc_dict = dict()
        self.tqdm_obj = tqdm_obj
        self.mininterval = mininterval
        self.maxinterval = maxinterval
        self.overall_timeout = overall_timeout
        self.instance_timeout = instance_timeout
        self.num_queued_tasks = 0
        self.task_index = 0
        self.start_time = None
        self.terminate_all = False
        if verbose:
            warnings.warn('''
            ParallelSubProcess is for tasks that are started with subprocess.Popen()
            The task should return the proc object that supports communicate
            For more details, see: https://docs.python.org/3/library/subprocess.html
            ''')
        #

    def run(self, task_entries):
        self.queued_tasks = task_entries
        self.num_queued_tasks = len(task_entries)
        self.task_index = 0

        self.start_time = time.time()
        desc = self.desc + f' TOTAL={self.num_queued_tasks}, NUM_RUNNING={0}'
        if self.tqdm_obj is None:
            self.tqdm_obj = tqdm.tqdm(total=self.num_queued_tasks, position=0, desc=desc)
        #

        completed_tasks = []
        # now search through tasks and find a process to start
        for task_name, task_list in self.queued_tasks.items():
            for proc_id, proc_dict in enumerate(task_list):
                proc_name = proc_dict['proc_name']

                running_tasks = [proc_name]
                num_running = len(running_tasks)
                desc = self.desc + f' TOTAL={self.num_queued_tasks}, NUM_RUNNING={num_running}'
                self.tqdm_obj.set_description(desc)
                self.tqdm_obj.set_postfix(postfix=dict(RUNNING=running_tasks, COMPLETED=completed_tasks))

                proc_dict['running'] = True
                proc_dict['completed'] = False
                proc_dict['proc'] = self._worker(proc_dict['proc_func'], proc_dict['proc_log'])
                if proc_dict['proc'] is not None:
                    proc_dict['proc'].communicate()
                #
                proc_dict['running'] = False
                proc_dict['completed'] = True
            #

            completed_tasks += [task_name]
            num_completed = len(completed_tasks)
            num_running = len(running_tasks)

            self.tqdm_obj.update(num_completed - self.tqdm_obj.n)
        #

    def _worker(self, task, log_file):
        proc = None
        try:
            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, 'a') as log_fp:
                    with wurlitzer.pipes(stdout=log_fp, stderr=wurlitzer.STDOUT):
                        result = task()
                    #
                #
            else:
                print(f"WARNING: log_file was not provided - running without capturing the log - {__file__}")
                result = task()
            #
        except KeyboardInterrupt:
            print(f"KeyboardInterrupt occurred in worker: {e}")
            traceback.print_exc()
            raise
        except Exception as e:
            print(f"Exception occurred in worker: {e}")
            traceback.print_exc()
        #
        self.task_index += 1
        return proc

    def terminate_all(self):
        pass
        