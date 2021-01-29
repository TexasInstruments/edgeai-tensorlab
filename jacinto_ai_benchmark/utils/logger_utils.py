import sys
import logging

class TeeLogger:
    def __init__(self, filename, log_level=logging.INFO, append=False):
        assert log_level == logging.INFO, 'for now we support only INFO logging level'
        mode = "a" if append else "w"
        self.term = sys.stdout
        self.file = open(filename, mode)
        sys.stdout = self
        self.count = 0

    def __del__(self):
        self.close()

    def close(self):
        if self.file is not None:
            sys.stdout = self.term
            self.file.close()
            self.file = None
        #

    def write(self, message):
        self.term.write(message)
        self.file.write(message)
        self.flush()

    def info(self, message):
        self.write(message)

    def debug(self, message):
        self.file.write(message)
        self.flush()

    def flush(self):
        self.term.flush()
        self.file.flush()

    def fileno(self):
        return self.term.fileno()
