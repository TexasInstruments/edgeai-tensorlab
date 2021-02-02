from multiprocessing import Process

class Parallel:
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.queued_tasks = []
        self.running_processes = [None for _ in range(self.num_processes)]

    def __del__(self):
        for p_idx, proc in enumerate(self.running_processes):
            if proc is not None:
                proc.terminate()
                proc.join()
            #
        #

    def enqueue(self, task):
        self.queued_tasks.append(task)

    def start(self):
        self._run()

    def wait(self):
        pass

    def _run(self):
        while (self._num_running() > 0) or (self._num_queued() > 0):
            if (self._num_running() < self.num_processes) and (self._num_queued() > 0):
                slot_idx = self._empty_slot()
                assert (slot_idx is not None), 'empty slot expected, but found none'
                task = self.queued_tasks.pop(0)
                proc = Process(target=task)
                self.running_processes[slot_idx] = proc
                proc.start()
            #
            for slot_idx, proc in enumerate(self.running_processes):
                if (proc is not None) and (not proc.is_alive()):
                    self.running_processes[slot_idx] = None
                    proc.join()
                #
            #
        #

    def _num_queued(self):
        return len(self.queued_tasks)

    def _num_running(self):
        return sum([1 if proc is not None else 0 for proc in self.running_processes])

    def _empty_slot(self):
        for p_idx, proc in enumerate(self.running_processes):
            if proc is None:
                return p_idx
            #
        #
        return None