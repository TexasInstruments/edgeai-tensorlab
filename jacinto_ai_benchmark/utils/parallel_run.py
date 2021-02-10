from multiprocessing import Process, JoinableQueue
from collections import deque
import time
import atpbar


class ParallelRun:
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.queued_tasks = deque()

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

    def worker(self, reporter, queue):
        atpbar.register_reporter(reporter)
        while True:
            task = queue.get()
            if task is None:
                queue.task_done()
                break
            else:
                task()
            #
            queue.task_done()
        #

    def _run_sequential(self):
        for task_id, task in atpbar.atpbar(self.queued_tasks, name='tasks'):
            task()
        #

    def _run_parallel(self):
        reporter = atpbar.find_reporter()
        queue = JoinableQueue()
        # start the processes
        for proc_id in range(self.num_processes):
            proc = Process(target=self.worker, args=(reporter, queue))
            proc.start()
        #
        # provide the processes something to chew
        for task in self.queued_tasks:
            queue.put(task)
        #
        # monitor the progress
        pbar = atpbar.atpbar(range(len(self.queued_tasks)), name='tasks')
        pbar = iter(pbar)
        qsize_prev = len(self.queued_tasks)
        num_next = 0
        while queue.qsize() > 0:
            qsize = queue.qsize()
            for step in range(qsize, qsize_prev):
                next(pbar)
                num_next += 1
            #
            qsize_prev = queue.qsize()
            time.sleep(1.0)
        #
        for step in range(num_next, len(self.queued_tasks)):
            next(pbar)
            num_next += 1
        #
        # join the process as they finish
        for proc_id in range(self.num_processes):
            queue.put(None)
            queue.join()
        #
        atpbar.flush()
