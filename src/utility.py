import atexit
from time import time

time_data = {}
def timer(f):
    def func(*args, **kwargs):
        t1 = time()
        res = f(*args, **kwargs)
        t2 = time()
        if f.__name__ not in time_data:
            time_data[f.__name__] = 0
        time_data[f.__name__] += t2 - t1
        return res
    return func

@atexit.register
def print_times():
    for k, v in time_data.items():
        print(k.ljust(25, ' '), v)
