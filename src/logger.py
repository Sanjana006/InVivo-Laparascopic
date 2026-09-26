import os
import sys
import logging

LOG_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline_run.log")
)

class TeeOutput:
    """Redirects writes to both original stream (stdout/stderr) and log file."""
    def __init__(self, original_stream, file_handle):
        self.original_stream = original_stream
        self.file_handle = file_handle

    def write(self, data):
        try:
            self.original_stream.write(data)
        except (UnicodeEncodeError, AttributeError):
            encoding = getattr(self.original_stream, "encoding", "utf-8") or "utf-8"
            safe_data = data.encode(encoding, errors="replace").decode(encoding, errors="replace")
            self.original_stream.write(safe_data)

        if self.file_handle and not self.file_handle.closed:
            self.file_handle.write(data)
            self.file_handle.flush()

    def flush(self):
        try:
            self.original_stream.flush()
        except Exception:
            pass
        if self.file_handle and not self.file_handle.closed:
            self.file_handle.flush()

    def isatty(self):
        return getattr(self.original_stream, "isatty", lambda: False)()

def setup_logger(name="pipeline", log_file=LOG_FILE_PATH, level=logging.INFO):
    """Sets up a logger that logs to both console and log_file without duplication."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File Handler
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream Handler (console using raw stdout to avoid Tee duplication)
        stream_handler = logging.StreamHandler(sys.__stdout__)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

def enable_output_redirection(log_file=LOG_FILE_PATH):
    """Redirects stdout and stderr to both stdout/stderr and log_file."""
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    file_handle = open(log_file, "a", encoding="utf-8")
    sys.stdout = TeeOutput(sys.__stdout__, file_handle)
    sys.stderr = TeeOutput(sys.__stderr__, file_handle)
    return file_handle


