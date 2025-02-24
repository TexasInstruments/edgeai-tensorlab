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


class ParallelRunner:
    def __init__(self, parallel_processes, desc='TASKS', mininterval=1.0, maxinterval=60.0, tqdm_obj=None,
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
        self.terminate_all_flag = False
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
        num_completed, num_running = self._wait_in_loop(self.parallel_processes)
        while num_completed < self.num_queued_tasks and (not self.terminate_all_flag):
            # wait in a loop until the number of running processes comes down
            num_completed, num_running = self._wait_in_loop(self.parallel_processes)
            proc_dict_to_start = self._find_proc_dict_to_start()
            # start the process
            if proc_dict_to_start and (not self.terminate_all_flag):
                proc_dict_to_start['running'] = True
                proc_dict_to_start['completed'] = False
                proc_dict_to_start['start_time'] = time.time()
                proc_dict_to_start['proc'] = self._worker(proc_dict_to_start['proc_func'])
            #
        #
        # wait for all remaining processes to finish
        if not self.terminate_all_flag:
            self._wait_in_loop(0)
        #
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
        if proc is not None:
            completed = False
            exit_code = proc.returncode
            try:
                err_code = proc.wait(timeout=0.1)
                if err_code:
                    raise subprocess.CalledProcessError(err_code, "Error occurred")
                #
            except subprocess.TimeoutExpired as ex:
                pass
            except multiprocessing.TimeoutError as ex:
                pass
            else:
                out_ret, err_ret = proc.communicate(timeout=0.1)
                completed = True
            #
        else:
            # proc = None indicates a completed task
            # especially happens if a ParallelProcess is not lanched, but is a simple task that returns None as proc.
            completed = True
        #
        return completed

    def _check_running_status(self, check_errors):
        running_tasks = []
        completed_tasks = []
        for task_name, task_list in self.queued_tasks.items():
            completed_proc_in_task = []
            running_proc_in_task = []
            running_proc_name = None
            for proc_id, proc_dict in enumerate(task_list):
                proc = proc_dict.get('proc', None)
                completed = proc_dict.get('completed', False)
                running = proc_dict.get('running', False)
                terminated = proc_dict.get('terminated', False)

                # check running processes                         
                if running:
                    # try to update the completed status for running processes
                    completed = self._check_proc_complete(proc)
                    running = (not completed)
                    proc_dict['completed'] = completed
                    proc_dict['running'] = running
                    running_proc_name = proc_dict['proc_name']
                    running_time = time.time() - proc_dict['start_time']
                    proc_log = proc_dict['proc_log']
                    proc_error = proc_dict['proc_error']
                    proc_error = [proc_error] if not isinstance(proc_error, (list,tuple)) else proc_error

                    # look for processes to terminate forcefully
                    # proc_dict entry of terminated process will eventually get removed - don't keep trying  to terminate it again and again    
                    if check_errors and (not terminated):
                        proc_terminate = False
                        proc_term_msgs = []
                        if proc_log and os.path.exists(proc_log):
                            with open(proc_log, "r") as fp:
                                try:
                                    proc_log_content = fp.read()
                                    for proc_error_entry in proc_error:
                                        regex_match =  re.search(proc_error_entry, proc_log_content)
                                        if regex_match:
                                            proc_terminate = True
                                            proc_term_msgs += [regex_match.group()]
                                        #
                                    #
                                except:
                                    print(f"WARNING: could not read file: {proc_log}")
                                #
                            #
                        #
                        if self.instance_timeout and running_time > self.instance_timeout and proc is not None:
                            proc_terminate = True
                            proc_term_msgs += [f"TIMEOUT : {self.instance_timeout}"]
                        #
                        if proc_terminate:
                            print(f"WARNING: terminating the process - {running_proc_name} - {', '.join(proc_term_msgs)}")
                            proc.terminate()
                            proc_dict['terminated'] = True
                        #
                    #
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
        desc = self.desc + f' TOTAL={self.num_queued_tasks}, NUM_RUNNING={num_running}'
        self.tqdm_obj.set_description(desc)
        self.tqdm_obj.set_postfix(postfix=dict(RUNNING=running_tasks, COMPLETED=completed_tasks))
        return num_completed, num_running

    def _wait_in_loop(self, num_processes):
        # wait in a loop until the number of running processes come down        
        num_processes = num_processes or 0
        last_check_time = time.time()        
        check_errors = True
        num_completed, num_running = self._check_running_status(check_errors=check_errors)
        while num_running > 0 and num_running >= num_processes and (not self.terminate_all_flag):
            num_completed, num_running = self._check_running_status(check_errors=check_errors)
            if num_running >= num_processes:
                time.sleep(self.maxinterval)
            #
            # check if this run has been too long; terminate if needed
            running_time = time.time() - self.start_time
            if self.overall_timeout and (running_time > self.overall_timeout):
                self.terminate_all()
            #
            check_interval = time.time() - last_check_time            
            check_errors = (check_interval > self.maxinterval)
            if check_errors:              
                last_check_time = time.time()          
            #
        #
        time.sleep(self.mininterval)        
        # check if this run has been too long; terminate if needed
        running_time = time.time() - self.start_time
        if self.overall_timeout and (running_time > self.overall_timeout) and (not self.terminate_all_flag):
            self.terminate_all()
        #        
        return num_completed, num_running

    def _terminate_all_procs(self, term_mesage=None):
        term_mesage = term_mesage or f"TIMEOUT - TERMINATE ALL: {self.overall_timeout}"
        for task_name, task_list in self.queued_tasks.items():
            for proc_id, proc_dict in enumerate(task_list):
                proc = proc_dict.get('proc', None)
                running_proc_name = proc_dict['proc_name']                
                running = (proc is not None) and proc_dict.get('running', False)
                terminated = (proc is not None) and proc_dict.get('terminated', False)                   
                if running and (not terminated):
                    proc.terminate()
                    proc_dict['terminated'] = True       
                    print(f"WARNING: terminating the process - {running_proc_name} - {term_mesage}")                                                   
                #
            #
        #     

    def _worker(self, task):
        # in ParallelRunner, we cannot capture the log here
        # It has to be captured inside the worker of the process - see the class utils.ProcessWithQueue
        proc = None
        try:
            proc = task()
        except KeyboardInterrupt:
            print(f"KeyboardInterrupt occurred: {__file__}")
            traceback.print_exc()
            raise
        except Exception as e:
            print(f"Exception occurred: {__file__}")
            traceback.print_exc()
        #
        self.task_index += 1
        return proc

    def terminate_all(self, term_mesage):
        self._terminate_all_procs(term_mesage)
        self.terminate_all_flag = True