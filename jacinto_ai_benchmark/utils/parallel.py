from multiprocessing import Process

class Parallel:
    def __init__(self):
        self.queued_processes = []
        self.running_processes = []


    def __del__(self):
        while len(self.running_processes)>0:
            p = self.running_processes.pop(0)
            p.terminate()
            p.join()
        #


    def queue(self, task):
        proc = Process(target=task)
        self.queued_processes.append(proc)


    def start(self):
        for p in self.queued_processes:
            p.start()
            self.running_processes.append(p)
        #


    def wait(self):
        while len(self.running_processes)>0:
            p = self.running_processes.pop(0)
            p.join()
        #