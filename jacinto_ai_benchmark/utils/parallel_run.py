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

    def _run_monitor(self, result_iterator, num_tasks, timeout=60.0, verbose=True):
        num_completed = 0
        time_taken_str = ''
        eta_str = ''
        it_per_sec = 0
        result_list = []
        start_time = time.time()
        while num_completed < num_tasks:
            try:
                result = result_iterator.__next__(timeout=timeout)
                result_list.append(result)
                num_completed += 1
                end_time = time.time()
                time_taken = (end_time - start_time)
                time_taken_str = self._delta_time_string(time_taken)
                eta = (time_taken / num_completed) * (num_tasks - num_completed)
                eta_str = self._delta_time_string(eta)
                it_per_sec = (num_completed / time_taken) if time_taken != 0 else 0.0
            except KeyboardInterrupt as e:
                logging.error(traceback.format_exc())
                return result_list
            except ValueError as e:
                logging.error(traceback.format_exc())
                return result_list
            except Exception as e:
                pass
            #
            if verbose:
                print(f" {Fore.MAGENTA}tasks: {num_completed}/{num_tasks} |"
                      f" {Fore.YELLOW}[{time_taken_str}<{eta_str},{it_per_sec:6.2f}it/s]{Fore.RESET}",
                      end=None)
            #
        #
        return result_list

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

