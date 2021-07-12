"""

"""

import os
import numpy as np
from icecream import ic  # better printing for debugging

from spider_bot.environments import SpiderBotSimulator

class Driver:
    def __init__(self) -> None:
        self.cwd = os.getcwd()
        self.env = SpiderBotSimulator(os.path.join(self.cwd, 'urdfs', 'spider_bot_v0.urdf'), real_time=False, gui=True)
    
    def run(self, args: list) -> None:
        self.showcase()
        
    def showcase(self) -> None:
        i = 0
        j = 0
        done = False
        while j < 3:
            controls = [-5 * np.sin(i / 10) for _ in range(4)]
            observation, reward, done, info = self.env.step(controls)
            pos, vel = observation['pos'], observation['vel']  # break down state of joints
            # log pos and vel
            i += 1
            #if done or i > 500:
            #    self.env.reset()
            #    j += 1
            #    i = 0
        self.env.close()
        
    def train(self):
        pass
    
    def graph_data(self):
        pass
    
    def save_model(self):
        pass
    
    def load_model(self):
        pass