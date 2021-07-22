"""

"""

import os
import sys
from time import sleep
from icecream import ic
from matplotlib import pyplot as plt

cwd = os.getcwd()
if cwd.split('\\') == 'Spider-Bot':
    sys.path.append(cwd)
else:
    parent = os.path.dirname(cwd)
    sys.path.append(parent)
    
from spider_bot import utils

def animation_cb(frame, axes, data, queue):
    if not queue.empty():
        new_data = queue.get()
        data.append(new_data)
    n = len(data)
    axes[0].plot(range(n), data, 'ro')

def make_fig():
    return plt.subplots(2, sharex=True)

def main():
    ic(os.getpid())
    plotter = utils.LivePlotter(animation_cb, make_fig)
    plotter.start()
    for i in range(10):
        plotter.send_data(i**2)
        sleep(1)
    plotter.close()

if __name__ == '__main__':
    main()