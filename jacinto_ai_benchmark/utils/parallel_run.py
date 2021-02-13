import sys
import multiprocessing
import collections
import time
from colorama import Fore
from .progress_step import *


class ParallelRun:
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.queued_tasks = collections.deque()
        self.results_list = []
        self.start_time = 0
        self.monitor_step = 60.0

    def enqueue(self, task):
        self.queued_tasks.append(task)

    def run(self):
        assert len(self.queued_tasks) > 0, f'at least one task must be queued, got {len(self.queued_tasks)}'
        self.start_time = time.time()
        return self._run_parallel()

    def _run_sequential(self):
        result_list = []
        for task_id, task in progress_step(self.queued_tasks, desc='tasks'):
            result = task()
            result_list.append(result)
        #
        return result_list

    def _run_parallel(self):
        # create process pool and queue the tasks
        process_pool = multiprocessing.Pool(self.num_processes)
        for task in self.queued_tasks:
            process_pool.apply_async(task, callback=self._record_result)
        #
        # monitor the progress
        result_list = self._run_monitor()
        return result_list

    def _record_result(self, result):
        self.results_list.append(result)

    def _run_monitor(self):
        it_per_sec = 0.0
        time_taken_str = eta_str = ''
        num_tasks = len(self.queued_tasks)
        monitor_step_completed = 0
        while len(self.results_list) < num_tasks:
            end_time = time.time()
            num_completed = len(self.results_list)
            delta_time = end_time - self.start_time
            monitor_step_new = delta_time//self.monitor_step
            if num_completed > 0:
                time_taken_str = self._delta_time_string(delta_time)
                eta_str = self._delta_time_string(delta_time*(num_tasks-num_completed)/num_completed)
                it_per_sec = (num_completed / delta_time)
            #
            if monitor_step_new > monitor_step_completed:
                print(f" {Fore.RED}tasks: {num_completed}/{num_tasks} |"
                      f" {Fore.YELLOW}[{time_taken_str}<{eta_str} {it_per_sec:5.2f}it/s]{Fore.RESET}",
                      end=None)
                monitor_step_completed = monitor_step_new
            #
            time.sleep(1.0)
        #
        return self.results_list

    def _delta_time_string(self, seconds):
        days = int(seconds//(60*60*24))
        seconds = int(seconds%(60*60*24))
        hours = int(seconds//(60*60))
        seconds = int(seconds%(60*60))
        minutes = int(seconds//(60))
        return f'{days}d,{hours:02d}:{minutes:02d}' if days > 0 \
            else f'{hours:02d}:{minutes:02d}'
