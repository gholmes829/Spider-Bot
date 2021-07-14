"""

"""

import os
import numpy as np
from icecream import ic  # better printing for debugging

from spider_bot.environments import SpiderBotSimulator
from spider_bot.agent import Agent

class Driver:
    def __init__(self) -> None:
        self.cwd = os.getcwd()
        self.spider_urdf_path = os.path.join(self.cwd, 'urdfs', 'spider_bot_v0.urdf')
        self.env = SpiderBotSimulator(self.spider_urdf_path, real_time_enabled=True, gui=True)
        self.agent = Agent(24, 12)
    
    def run(self, args: list) -> None:
        self.episode()
        
    def episode(self) -> None:
        i = 0
        done = False

        observation = self.env.reset()
        controls = self.agent.predict(observation)
        
        done_msg_sent = False
        while not done or True:
            observation, reward, done, info = self.env.step(controls)
            if done and not done_msg_sent:
                ic(f'Normally episode would terminate now (i={i})...')
                done_msg_sent = True
            self.log_state(observation, controls)
            
            controls = self.agent.predict(self.preprocess(observation))
            i += 1

        self.env.close()
        
    def preprocess(self, observation):
        pos, vel = np.split(observation, [12])
        normal_pos = pos / (2 * np.pi)
        normal_vel = vel / self.env.spider.nominal_joint_velocity
        return np.array([*normal_pos, *normal_vel])
        
    def log_state(self, observation, controls):
        pass
        
    def train(self):
        pass
    
    def save_model(self):
        pass
    
    def load_model(self):
        pass