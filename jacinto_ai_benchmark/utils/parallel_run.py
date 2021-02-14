import sys
import multiprocessing
import collections
import time
from colorama import Fore
from .progress_step import *
from .timer_utils import delta_time_string


class ParallelRun:
    def __init__(self, num_processes, desc='tasks', blocking=True, maxinterval=60.0):
        self.desc = desc
        self.num_processes = num_processes
        self.queued_tasks = collections.deque()
        self.maxinterval = maxinterval
        self.blocking = blocking

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
        results_iterator = process_pool.imap_unordered(self._worker, self.queued_tasks)
        if self.blocking:
            # run a loop to monitor the progress
            result_list = self._run_monitor(results_iterator)
            return result_list
        else:
            return results_iterator
        #

    def _run_monitor(self, results_iterator):
        results_list = []
        it_per_sec = 0.0
        num_completed = 0
        time_taken_str = eta_str = ''
        num_tasks = len(self.queued_tasks)
        start_time = time.time()
        while num_completed < num_tasks:
            # check if a result is available
            try:
                result = results_iterator.__next__(timeout=self.maxinterval)
                results_list.append(result)
                num_completed = len(results_list)
            except multiprocessing.TimeoutError as e:
                pass
            #
            # estimate the arrival time
            if num_completed > 0:
                end_time = time.time()
                delta_time = end_time - start_time
                time_taken_str = delta_time_string(delta_time)
                eta_str = delta_time_string(delta_time*(num_tasks-num_completed)/num_completed)
                it_per_sec = (num_completed / delta_time)
            #
            # display the progress
            display_str = f"\r {Fore.RED}{self.desc}: {num_completed}/{num_tasks} |" \
                          f" {Fore.YELLOW}[{time_taken_str}<{eta_str} {it_per_sec:5.2f}it/s]{Fore.RESET}"
            print(display_str, end='')
            sys.stdout.flush()
        #
        return results_list

    def _worker(self, task):
        return task()