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

def GraphJointVelocities(velocities: np.array, location: str) -> plt.Axes:
    """ 
    Creates four 2D graphs of the velocities of each of the 
    robots joints over time. 

    """

    front_left, front_right, back_left, back_right = velocities[0:4]
    x = np.linspace(0, front_left.size, num=front_left.size)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(location + " Joint Velocities")

    axs[0, 0].plot(x, front_left)
    axs[0, 0].set_title('Front Left')
    axs[0, 0].set_ylim(-3, 3)

    axs[0, 1].plot(x, front_right, 'tab:orange')
    axs[0, 1].set_title('Front Right')
    axs[0, 1].set_ylim(-3, 3)

    axs[1, 0].plot(x, back_left, 'tab:green')
    axs[1, 0].set_title('Back Left')
    axs[1, 0].set_ylim(-3, 3)

    axs[1, 1].plot(x, back_right, 'tab:red')
    axs[1, 1].set_title('Back Right')
    axs[1, 1].set_ylim(-3, 3)
    
    return axs

