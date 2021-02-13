import multiprocessing
import collections
import time
import traceback
import logging
from colorama import Fore
from .progress_step import *


class ParallelRun:
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.queued_tasks = collections.deque()

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
        num_tasks = len(self.queued_tasks)
        # create process pool and queue the tasks
        process_pool = multiprocessing.Pool(self.num_processes)
        results = process_pool.imap_unordered(self._worker, self.queued_tasks)
        # monitor the progress
        result_list = self._run_monitor(results, num_tasks)
        return result_list

    def _run_monitor(self, results_iterator, num_tasks):
        results_list = []
        for result in progress_step(results_iterator, total=num_tasks, desc=' tasks: '):
            results_list.append(result)
        #
        return results_list

    def _delta_time_string(self, seconds):
        days = int(seconds//(60*60*24))
        seconds = int(seconds%(60*60*24))
        hours = int(seconds//(60*60))
        seconds = int(seconds%(60*60))
        minutes = int(seconds//(60))
        # seconds = int(seconds%(60))
        return f'{days}d,{hours:02d}:{minutes:02d}'

    def _worker(self, task):
        result = task()
        return result

