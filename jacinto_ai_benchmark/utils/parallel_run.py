from multiprocessing import Process
from collections import deque
import progiter


class ParallelRun:
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.queued_tasks = deque()
        self.running_processes = [None for _ in range(self.num_processes)]
        self.total_queued = 0

    def __del__(self):
        for proc_idx, proc in enumerate(self.running_processes):
            if proc is not None:
                proc.terminate()
                proc.join()
            #
        #

    def enqueue(self, task):
        self.queued_tasks.append(task)

    def start(self):
        self.total_queued = len(self.queued_tasks)
        assert self.total_queued > 0, f'atleast one task must be queued, got {self.total_queued}'
        if self.total_queued == 1:
            task = self.queued_tasks.popleft()
            task()
        else:
            self._run_parallel()
        #

    def wait(self):
        pass

    def _run_parallel(self):
        progress_bar = progiter.ProgIter(desc='running models: ', total=self.total_queued, verbose=1)
        progress_bar.begin()
        while (self._num_running() > 0) or (self._num_queued() > 0):
            if (self._num_running() < self.num_processes) and (self._num_queued() > 0):
                slot_idx = self._empty_slot()
                assert (slot_idx is not None), 'empty slot expected, but found none'
                task = self.queued_tasks.popleft()
                proc = Process(target=task)
                self.running_processes[slot_idx] = proc
                proc.start()
            #
            for slot_idx, proc in enumerate(self.running_processes):
                if (proc is not None) and (not proc.is_alive()):
                    self.running_processes[slot_idx] = None
                    proc.join()
                    progress_bar.step(1)
                #
            #
        #
        progress_bar.close()

    def _num_queued(self):
        return len(self.queued_tasks)

    def _num_running(self):
        return sum([1 if proc is not None else 0 for proc in self.running_processes])

    def _empty_slot(self):
        for proc_idx, proc in enumerate(self.running_processes):
            if proc is None:
                return proc_idx
            #
        #
        return None