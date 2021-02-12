import multiprocessing
import collections
from .progress_indicator import *


class ParallelRun:
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.queued_tasks = collections.deque()

    def enqueue(self, task):
        self.queued_tasks.append(task)

    def start(self):
        assert len(self.queued_tasks) > 0, f'at least one task must be queued, got {len(self.queued_tasks)}'
        if len(self.queued_tasks) == 1 or self.num_processes <= 1:
            self._run_sequential()
        else:
            self._run_parallel()
        #

    def wait(self):
        pass

    def _run_sequential(self):
        for task_id, task in progress_indicator(self.queued_tasks, desc='tasks'):
            task()
        #

    def _worker(self, task):
        result = task()
        return result

    def _run_parallel(self):
        num_tasks = len(self.queued_tasks)
        # create process pool and queue the tasks
        process_pool = multiprocessing.Pool(self.num_processes)
        results = process_pool.imap_unordered(self._worker, self.queued_tasks)
        # results does not have len - explicitly give it, so that tqdm can use it
        results = IterableWithLength(results, num_tasks)
        # monitor the progress
        for result in progress_indicator(results, desc='tasks'):
            pass
        #


class IterableWithLength:
    def __init__(self, iterable, length):
        self.iterable = iterable
        self.length = length

    def __iter__(self):
        return self.iterable.__iter__()

    def __len__(self):
        return self.length