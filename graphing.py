import matplotlib.pyplot as plt
import numpy as np

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
            axs[i, j].plot(x, data[index], colors[index])
            axs[i, j].set_title(subtitles[index])
            axs[i, j].set_yticks([0, 1])

    # axs[0, 0].plot(x, front_left, 'tab:orange')
    # axs[0, 0].set_title('Front Left')
    # #axs[0, 0].set_ylim(0, 1)

    # axs[0, 1].plot(x, front_right, 'tab:green')
    # axs[0, 1].set_title('Front Right')
    # #axs[0, 1].set_ylim(0, 1)

    # axs[1, 0].plot(x, back_left, 'tab:olive')
    # axs[1, 0].set_title('Back Left')
    # #axs[1, 0].set_ylim(, 3)

    # axs[1, 1].plot(x, back_right, 'tab:blue')
    # axs[1, 1].set_title('Back Right')
    # #axs[1, 1].set_ylim(-3, 3)
    
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
        ymax = data[np.where(data == data.max())] * 1.05
    if ymin is None:
        ymin = data[np.where(data == data.min())] * 1.05

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