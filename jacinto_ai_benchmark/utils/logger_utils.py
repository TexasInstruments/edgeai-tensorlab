import os
import sys


class TeeLogger:
    def __init__(self, file, stream=sys.stdout, with_tee=True):
        super().__init__()
        self.file = file
        self.stream = stream
        self.with_tee = with_tee
        self.is_open = False
        self._start()

    def __del__(self):
        self.close()

    def close(self):
        if self.is_open:
            self._finish()
        #

    def flush(self):
        self.stream.flush()
        if not self.file.closed:
            self.file.flush()
        #

    def write(self, message):
        self.stream.write(message)

    def info(self, message):
        self.write(message)

    def debug(self, message):
        self.write(message)

    def fileno(self):
        return self.file.fileno()

    def _start(self):
        # save a copy of stdout
        self.stdout_fileno_orig = self.stream.fileno()
        # duplicate stdout if needed
        if self.with_tee:
            self.stdout_fileno_new = os.dup(self.stdout_fileno_orig)
        else:
            self.stdout_fileno_new = self.stdout_fileno_orig
        #
        # redirect the stdout duplicate
        os.dup2(self.file.fileno(), self.stdout_fileno_new)
        self.is_open = True

    def _finish(self):
        # reset stdout
        self.file.flush()
        self.stream.flush()
        if self.with_tee:
            os.close(self.stdout_fileno_new)
        else:
            os.dup2(self.stdout_fileno_new, self.stdout_fileno_orig)
        #
        self.is_open = False
