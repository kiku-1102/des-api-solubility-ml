"""Plain text logger: every run writes an execution.log next to its results."""

from pathlib import Path


class Logger:
    """Write a message to stdout and to a log file at the same time."""

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, "w", encoding="utf-8")

    def log(self, msg: str = "", end: str = "\n"):
        print(msg, end=end, flush=True)
        self.log_file.write(msg + end)
        self.log_file.flush()

    def close(self):
        self.log_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
