import sys


class Logger(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


def create_logger(directory, name):
    logfile = directory + name + ".txt"
    print(f"Logging outputs at {logfile}")

    # original_stdout = sys.stdout
    log = open(logfile, "a", encoding="utf-8")
    file_logger = Logger(sys.stdout, log)

    sys.stdout = file_logger
    sys.stderr = file_logger
