"""

"""

import os
from typing import Callable, Any
from matplotlib import pyplot as plt
from matplotlib import animation
import multiprocessing as mp
from functools import wraps
from time import time

def timed(f):
    @wraps(f)
    def g(*args, **kwargs):
        start = time()
        out = f(*args, **kwargs)
        elapsed = time() - start
        return out, elapsed
    return g

def simplify_anim_cb_signature(f):
    @wraps(f)
    def anim_cb(_, *args, **kwargs):
        out = f(*args, **kwargs)
        return out
    return anim_cb

class LivePlotter:
    def __init__(self, animation_cb, initialize_axes, verbose=True, **kwargs) -> None:
        self.queue = mp.Queue()
        self.process = mp.Process(target=self.run, args=(self.queue, animation_cb, initialize_axes, verbose), kwargs=kwargs)
        
    def start(self):
        self.process.start()
        
    def send_data(self, data: Any):
        self.queue.put(data)
        
    @staticmethod
    def run(queue, animation_cb: Callable, initialize_axes: Callable, verbose, **kwargs):
        if verbose: print(f'Async plotter spawned with PID = {os.getpid()}', flush=True)
        memory = {}
        fig, axes = initialize_axes(memory)
        anim  = animation.FuncAnimation(fig, animation_cb, fargs=(axes, memory, queue), **kwargs)
        plt.show()
        
    def close(self):
        self.queue.close()
        self.process.join()
        self.process.close()