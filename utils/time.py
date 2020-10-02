from datetime import timedelta, datetime
from prettytable import PrettyTable
import time


class StopWatch(object):

    def __init__(self, start=True):

        self._t1 = None
        self._t2 = None

        if start:
            self.start()

    def start(self):
        self._t1 = time.time()

    def stop(self):
        self._t2 = time.time()

    def runtime(self):
        return self._t2 - self._t1

    def runtime_str(self):
        return str(timedelta(seconds=self.runtime()))


class Timer(object):

    def __init__(self, NumSteps):

        self._start = datetime.now()
        self._t1 = time.time()
        self._NumSteps = NumSteps

        self._stop_stime = None

        self._threads = dict()
        self._thread_start_time = None
        self._active_thread = None

    def __call__(self, thread):

        if thread not in self._threads:
            self._threads[thread] = 0

        self._active_thread = thread
        self._thread_start_time = time.time()

        return self


    def _rrt(self, step):

        if step == 0:
            step = 0.0001

        fraction = step / self._NumSteps
        curr_runtime = time.time() - self._t1
        projected_runtime = (1/fraction) * curr_runtime
        remaining_runtime = projected_runtime - curr_runtime
        return remaining_runtime

    def stop(self):

        self._stop_time = datetime.now()

    def RRT(self, step, verbose = False):

        td = timedelta(seconds=self._rrt(step))
        runtime = "{:d} Days, {:02d}h:{:02d}m:{:02d}s".format(td.days, td.seconds // 3600, (td.seconds//60)%60 , td.seconds % 60)

        if verbose:
            print("Estimated Remaining runtime: " + runtime)

        return runtime

    def ETA(self, step):

        eta = timedelta(seconds=self._rrt(step)) + datetime.now()
        return eta.strftime('ETA: %d.%m.%Y, %H:%M:%S')

    def __enter__(self):
        self('default')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._threads[self._active_thread] += time.time() - self._thread_start_time
        self._active_thread = None

    def _construct_table(self, OrderByExecutionTime=True):

        T = PrettyTable()
        T.field_names = ["Job", "Runtime", "Fraction"]
        T.add_row(['Overall', datetime.now(), 1])

        for thread, runtime in self._threads.items():
            T.add_row([thread, runtime, runtime/10])

        return T


    def __str__(self):

        return self._construct_table().get_string()






