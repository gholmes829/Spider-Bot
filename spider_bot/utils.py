"""

"""

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
    def __init__(self, animation_cb, make_fig) -> None:
        self.queue = mp.Queue()
        self.process = mp.Process(target=self.run, args=(self.queue, animation_cb, make_fig))
        
    def start(self):
        self.process.start()
        
    def send_data(self, data: Any):
        self.queue.put(data)
        
    @staticmethod
    def run(queue, animation_cb: Callable, make_fig: Callable):
        fig, axes = make_fig()
        data = []
        anim  = animation.FuncAnimation(fig, animation_cb, fargs=(axes, data, queue))
        plt.show()
        
    def close(self):
        self.queue.close()
        self.process.join()
        self.process.close()