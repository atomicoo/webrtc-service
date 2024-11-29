import threading
import time


class TimerThread(threading.Thread):
    def __init__(self, duration, callback, **kwargs):
        super().__init__()
        self.duration = duration
        self.callback = callback
        self.kwargs = kwargs

    def run(self):
        time.sleep(self.duration)
        if self.callback:
            self.callback(**self.kwargs)


def run_a_timer(time_seconds, is_block, callback=None, **kwargs):
    if is_block:
        time.sleep(time_seconds)
        if callback:
            callback( **kwargs)
    else:
        timer = TimerThread(time_seconds, callback, **kwargs)
        timer.daemon = True
        timer.start()
