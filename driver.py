"""

"""

import os
import numpy as np
from icecream import ic  # better printing for debugging
import matplotlib.pyplot as plt
from spider_bot.environments import SpiderBotSimulator

class Driver:
    def __init__(self) -> None:
        self.cwd = os.getcwd()
        self.env = SpiderBotSimulator(os.path.join(self.cwd, 'urdfs', 'spider_bot_v0.urdf'), real_time_enabled=False, gui=True)
    
    def run(self, args: list) -> None:
        self.showcase()
        
    def showcase(self) -> None:
        i = 0
        done = False
        magnitude = 3.75
        alt = 0
        period = 100
        joint_positions = []
        joint_velocities = []
        body_positions = []

        # manual tuned controls
        while i < 500:
            controls = [
                # outer
                0.375 * (2 * alt - 1) * magnitude,  
                0.375 * (2 * int(not alt) - 1) * magnitude,
                0.375 * (2 * int(not alt) - 1) * magnitude,
                0.375 * (2 * alt - 1) * magnitude,
                # middle
                -0.1 * (2 * alt - 1) * magnitude,  
                -0.1 * (2 * int(not alt) - 1) * magnitude,
                -0.1 * (2 * int(not alt) - 1) * magnitude,
                -0.1 * (2 * alt - 1) * magnitude, 
                # inner
                0.5 * (2 * alt - 1) * magnitude,  
                0.5 * (2 * int(not alt) - 1) * magnitude,
                -0.5 * (2 * int(not alt) - 1) * magnitude,
                -0.5 * (2 * alt - 1) * magnitude,
            ]

            observation, reward, done, info = self.env.step(controls)
            pos, vel = observation['pos'], observation['vel']  # break down state of joints

            joint_positions.append(pos)
            joint_velocities.append(vel)
            body_positions.append(info['pos'])

            i += 1
            if i % period == 0:
                alt = int(not alt)

        self.graph_data(np.array(joint_positions).T, 
                        np.array(joint_velocities).T,
                        np.array(body_positions).T)
        self.env.close()
        
    def train(self):
        pass
    
    def graph_data(self, 
                   joint_positions:  np.array, 
                   joint_velocities: np.array,
                   body_positions:   np.array
                   ) -> None:

        plt.style.use(["dark_background"])
        #plt.rc("grid", linestyle="dashed", color="white", alpha=0.25)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(body_positions[0][0], body_positions[1][0], body_positions[2][0], c = 'blue', label = 'Start')
        ax.plot3D(body_positions[0], body_positions[1], body_positions[2], c = 'orange')
        ax.scatter3D(body_positions[0][-1], body_positions[1][-1], body_positions[2][-1], c = 'green', label = 'End')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc="upper left")
        ax.set_title("Body Position")
        #plt.show()
        #ax.grid()
        plt.savefig('body_position')
        print('here')
        #print("Body:", body_positions)
        #pass
    
    def save_model(self):
        pass
    
    def load_model(self):
        pass