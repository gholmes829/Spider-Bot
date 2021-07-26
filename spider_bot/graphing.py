"""

"""

import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np

from spider_bot import utils
 
@utils.simplify_anim_cb_signature
def live_training_cb(axes, memory: dict, queue: mp.Queue):
    if not queue.empty():
        new_data = queue.get()
        for key, item in new_data.items(): memory[key].append(item)
        
        average_fitness = memory['average_fitness']
        best_fitness = memory['best_fitness']
        time_elapsed = memory['time_elapsed']
        top_10_fitness = memory['top_10_fitness']
        
        assert len(average_fitness) == len(best_fitness) == len(time_elapsed)
        
        n = len(average_fitness)
        
        axes[0].plot(range(n), average_fitness, 'r')
        axes[0].plot(range(n), best_fitness, 'g')
        axes[0].plot(range(n), top_10_fitness, 'cyan')
        
        axes[1].plot(range(n), time_elapsed, 'magenta')
        
        axes[0].set_xlim([0, n + 2])
        axes[1].set_xlim([0, n + 2])
        
        axes[0].set_ylim([0, 1.33 * max(best_fitness)])
        axes[1].set_ylim([0, 1.33 * max(time_elapsed)])
    
def initialize_axes(memory: dict):
    plt.style.use('dark_background')
    memory['average_fitness'] = []
    memory['best_fitness'] = []
    memory['time_elapsed'] = []
    memory['top_10_fitness'] = []
    
    fig, axes = plt.subplots(2)
    
    axes[0].plot([], [], 'r', label='Average')
    axes[0].plot([], [], 'g', label='Best')
    axes[0].plot([], [], 'cyan', label='Top 10%')
    
    axes[1].plot([], [], 'magenta')
    
    axes[0].set_title('Fitnesses')
    axes[0].set_ylabel('Fitness')
    axes[0].grid(alpha=0.25, ls='--')
    axes[0].set_xlabel('Generations')
    axes[0].legend(loc="upper left")
    axes[0].set_xlim([0, 2])
    axes[0].set_ylim([0, 1])
    
    axes[1].set_title('Time Elapsed')
    axes[1].set_ylabel('Time (secs)')
    axes[1].grid(alpha=0.25, ls='--')
    axes[1].set_xlabel('Generations')
    axes[1].set_xlim([0, 2])
    axes[1].set_ylim([0, 1])

    plt.tight_layout()

    return fig, axes

def GraphBodyTrajectory(body_pos: np.array) -> plt.Axes:
    """
    Creates a 3D graph of the body's position over time,
    indicating the beginning and end of the trajectory.

    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(body_pos[0][0], body_pos[1][0], body_pos[2][0], c = 'blue', label = 'Start')
    ax.plot3D(body_pos[0], body_pos[1], body_pos[2], c = 'orange')
    ax.scatter3D(body_pos[0][-1], body_pos[1][-1], body_pos[2][-1], c = 'green', label = 'End')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left")
    ax.set_title("Body Position")
    return ax

def GraphContactData(data: np.array) -> plt.Axes:
    """ 
    Creates four 2D graphs of binary contact data
    for each leg

    """
    colors = ['tab:orange', 'tab:green', 'tab:olive', 'tab:blue']
    subtitles: list = ["Front Left", "Front Right", "Back Left", "Back Right"]

    front_left, front_right, back_left, back_right = data[0:4]
    x = np.linspace(0, front_left.size, num=front_left.size)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Leg Contact Over Time")

    for i in range(2):
        for j in range(2):
            index = 2 * i + j
            for k, (t, point) in enumerate(zip(x[:-1], data[index][:-1])):
                next_point = data[index][k + 1]

                if point + next_point == 0:
                    axs[i, j].plot([t, t + 1], [0, 0], colors[index])
                elif point + next_point == 2:
                    axs[i, j].plot([t, t + 1], [1, 1], colors[index])
                elif point == 1 and next_point == 0:  # rising edge
                    axs[i, j].plot([t, t + 1], [1, 1], colors[index])
                    axs[i, j].plot([t + 1, t + 1], [1, 0], colors[index])
                elif point == 0 and next_point == 1:  # falling edge
                    axs[i, j].plot([t, t + 1], [0, 0], colors[index])
                    axs[i, j].plot([t + 1, t + 1], [0, 1], colors[index])
                else:
                    raise ValueError(f'Invalid point, next point: {point, next_point}')
                #axs[i, j].plot(x, data[index], colors[index])
            axs[i, j].set_title(subtitles[index])
            axs[i, j].set_yticks([0, 1])
    
    return axs

def GraphJointData(data: np.array, 
                   title: str, 
                   subtitles: list = ["Inner Joints", "Middle Joints", "Outer Joints"],
                   ymin = None,
                   ymax = None
                   ) -> plt.Axes:
    """
    Creates 3 sets of subplots, each 2x2, of joint data.
    Each set contains all of the inner, middle, or outer joint data.

    """
    colors = ['tab:orange', 'tab:green', 'tab:olive', 'tab:blue']
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle(title + "\n")
    gs = fig.add_gridspec(2, 8, wspace=0, hspace=0.0)
    axs = gs.subplots()
    
    x = np.linspace(0, data[0].size, num=data[0].size)
    if ymax is None:
        ymax = data.max() * 1.05
    if ymin is None:
        ymin = data.min()

    index = 0
    for i in range(3):
        for j in range(2):
            for k in range(2):
                axs[j, 3 * i + k].plot(x, data[index], colors[2 * j + k], lw = .9)
                axs[j, 3 * i + k].set_ylim(ymin, ymax)
                axs[j, 3 * i + k].grid(color='dimgray', ls = '-', lw = .25)
                index += 1
            if k == 1:
                axs[j, 3 * i + k].yaxis.set_ticklabels([])
        axs[0, 3 * i].set_title(subtitles[i])
    

    for a in range(0, 2):
        for b in range(2, 6, 3):
            axs[a, b].remove()
    
    return axs

def GraphFitness(fitnesses: np.array) -> plt.Axes:
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(0, fitnesses.size, num=fitnesses.size)
    ax.plot(x, fitnesses)
    ax.set_ylim(0, fitnesses.max() * 1.05)
    ax.set_title("Fitness Over Time")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Average Fitness")
    return ax

def GraphAnkleHeights(data: np.array) -> plt.Axes:
    """ 
    Creates four 2D graphs of the heights of each ankle

    """
    colors = ['tab:orange', 'tab:green', 'tab:olive', 'tab:blue']
    subtitles: list = ["Front Left", "Front Right", "Back Left", "Back Right"]

    front_left, front_right, back_left, back_right = data[0:4]
    x = np.linspace(0, front_left.size, num=front_left.size)

    ymax = data.max() * 1.05
    ymin = data.min()

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Ankle Heights Over Time")

    for i in range(2):
        for j in range(2):
            index = 2 * i + j
            axs[i, j].plot(x, data[index], colors[index])
            axs[i, j].set_title(subtitles[index])
            axs[i, j].set_ylim(ymin, ymax)
    
    return axs
