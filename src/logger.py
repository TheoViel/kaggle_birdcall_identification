import sys


class Logger(object):
    """
    Simple logger that saves what is printed in a file
    """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def create_logger(directory="", name="logs.txt"):
    """
    Creates a logger to log output in a chosen file

    Keyword Arguments:
        directory {str} -- Path to save logs at (default: {''})
        name {str} -- Name of the file to save the logs in (default: {'logs.txt'})
    """
    logfile = directory + name + ".txt"
    print(f"Logging outputs at {logfile}")

    log = open(logfile, "a", encoding="utf-8")
    file_logger = Logger(sys.stdout, log)

    sys.stdout = file_logger
    sys.stderr = file_logger
