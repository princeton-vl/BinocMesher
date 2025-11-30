from datetime import datetime
import os

import psutil


class Timer:

    def __init__(self, desc, disable_timer=False, write_to_file=None):
        self.disable_timer = disable_timer
        self.write_to_file = write_to_file
        if self.disable_timer:    
            return
        self.name = f'[{desc}]'

    def __enter__(self):
        if self.disable_timer:
            return
        self.start = datetime.now()


    def __exit__(self, exc_type, exc_val, traceback):
        if self.disable_timer:
            return
        self.end = datetime.now()
        self.duration = self.end - self.start # timedelta
        if exc_type is None:
            process = psutil.Process(os.getpid())
            info = f'{self.name} finished in {str(self.duration)} with memory usage {process.memory_info().rss / 1024**3} GB'
        else:
            info = f'{self.name} failed with {exc_type}'
        if self.write_to_file is None:
            print(info)
        else:
            with open(self.write_to_file, "a") as f:
                f.write(f"{info}\n")
