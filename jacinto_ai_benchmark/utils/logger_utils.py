import os
import sys

class TeeLogger:
    def __init__(self, file_name):
        super().__init__()
        self.src_stream = sys.stdout
        self.dst_file = open(file_name, 'w') if file_name is not None else None
        sys.stdout = self

    def __del__(self):
        self.close()

    def write(self, message):
        if self.src_stream is not None:
            self.src_stream.write(message)
        #
        if self.dst_file is not None:
            self.dst_file.write(message)
        #
        self.flush()

    def flush(self):
        if self.src_stream is not None:
            self.src_stream.flush()
        #
        if self.dst_file is not None:
            self.dst_file.flush()
        #

    def isatty(self):
        return self.src_stream.isatty()

    def close(self):
        if self.src_stream is not None:
            sys.stdout = self.src_stream
            self.src_stream = None
        #
        if self.dst_file is not None:
            self.dst_file.close()
            self.dst_file = None
        #


class RedirectLogger:
    """
    Redirect the outputs of current process and subprocesses into a file
    """
    def __init__(self, file_name):
        super().__init__()
        self.src_stream = sys.stdout
        self.dst_file = open(file_name, 'w')
        # save a copy of stdout fd
        self.src_fd_orig = self.src_stream.fileno()
        self.src_fd_copy = os.dup(self.src_fd_orig)
        # connect file fd to stdout
        os.dup2(self.dst_file.fileno(), self.src_fd_orig)

    def __del__(self):
        self.close()

    def flush(self):
        self.src_stream.flush()
        if self.dst_file is not None:
            self.dst_file.flush()
        #

    def close(self):
        if self.dst_file is not None:
            self.dst_file.flush()
            self.src_stream.flush()
            # restore original stdout
            os.dup2(self.src_fd_copy, self.src_fd_orig)
            # cleanup
            os.close(self.src_fd_copy)
            self.dst_file.close()
            self.dst_file = None
        #

