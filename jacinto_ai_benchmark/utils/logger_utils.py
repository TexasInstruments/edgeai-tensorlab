import sys
import logging

class TeeLogger:
    def __init__(self, filename, log_level=logging.INFO, append=False):
        assert log_level == logging.INFO, 'for now we support only INFO logging level'
        mode = "a" if append else "w"
        self.term = sys.stdout
        self.termerr = sys.stderr
        self.filename = filename
        self.file = open(filename, mode)
        # avoid an error related to isatty
        # self.isatty = sys.stdout.isatty
        # self.encoding = sys.stdout.getdefaultencoding()
        #stdout and stderr to file and to term
        sys.stdout = self
        sys.stderr = self


    def __del__(self):
        self.flush()
        self.close()

    def close(self):
        if self.file is not None:
            sys.stdout = self.term
            sys.stderr = self.termerr
            self.file.close()
            self.file = None
        #

    def write(self, message):
        self.term.write(message)
        self.file.write(message)
        self.flush()

    def write_term(self, message):
        self.flush()
        self.term.write(message)
        self.flush()

    def write_file(self, message):
        self.flush()
        self.file.write(message)
        self.flush()

    def info(self, message):
        self.write(message)

    def debug(self, message):
        self.file.write(message)
        self.flush()

    def flush(self):
        self.term.flush()
        if self.file is not None:
            self.file.flush()
        #

    def fileno(self):
        return self.term.fileno()
