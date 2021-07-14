"""

"""

import os
import numpy as np
from icecream import ic  # better printing for debugging

from spider_bot.environments import SpiderBotSimulator
from spider_bot.agent import Agent
from spider_bot.training import Evolution

class Driver:
    def __init__(self) -> None:
        self.cwd = os.getcwd()
        self.spider_urdf_path = os.path.join(self.cwd, 'urdfs', 'spider_bot_v0.urdf')
        self.env = SpiderBotSimulator(self.spider_urdf_path, real_time_enabled=True, gui=True)
    
    def run(self, args: list) -> None:
        self.episode(Agent(24, 12))
        
    def preprocess(self, observation):
        pos, vel = np.split(observation, [12])
        normal_pos = pos / (2 * np.pi)
        normal_vel = vel / self.env.spider.nominal_joint_velocity
        return np.array([*normal_pos, *normal_vel])
        
    def episode(self, agent, logging=False) -> None:
        i = 0
        max_steps = 5096
        done = False
        rewards = []
        observation = self.env.reset()
        controls = agent.predict(observation)
        
        try:
            while not done and i < max_steps:
                observation, reward, done, info = self.env.step(controls)
                rewards.append(reward)
                if logging:
                    self.log_state(observation, controls)
                
                controls = agent.predict(self.preprocess(observation))
                i += 1
        except KeyboardInterrupt:
            self.env.close()
        return rewards
        
    def calc_fitness(agent):
        return 0
        
    def log_state(self, observation, controls):
        pass
        
    def train(self):
        pass
    
    def save_model(self):
        pass
    
    def load_model(self):
        pass