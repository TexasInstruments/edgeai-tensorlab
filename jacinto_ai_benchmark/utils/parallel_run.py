import sys
import multiprocessing
import collections
import time
from .progress_step import *


class ParallelRun:
    def __init__(self, num_processes, desc='tasks', blocking=True, maxinterval=10.0):
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
        num_completed = num_completed_prev = 0
        num_tasks = len(self.queued_tasks)
        pbar_tasks = progress_step(iterable=range(num_tasks), desc=self.desc, position=1)
        while num_completed < num_tasks:
            # check if a result is available
            try:
                result = results_iterator.__next__(timeout=self.maxinterval)
                results_list.append(result)
                num_completed = len(results_list)
            except multiprocessing.TimeoutError as e:
                pass
            #
            pbar_tasks.update(num_completed-num_completed_prev)
            num_completed_prev = num_completed
        #
        pbar_tasks.close()
        print('\n')
        return results_list

    def _worker(self, task):
        return task()