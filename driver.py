"""

"""

import os
import numpy as np
from icecream import ic  # better printing for debugging
import argparse

from spider_bot.environments import SpiderBotSimulator
from spider_bot.agent import Agent
from spider_bot.training import Evolution

class Driver:
    def __init__(self):
        self.modes = {
            'test': lambda: self.episode(Agent(24, 12)),
            'train': lambda: self.train()
        }
        
        parser = argparse.ArgumentParser()
        parser.add_argument('mode', choices = self.modes.keys(), help = 'determines which mode to run')
        args = parser.parse_args()
        self.mode = args.mode
        
        self.cwd = os.getcwd()
        self.spider_urdf_path = os.path.join(self.cwd, 'urdfs', 'spider_bot_v0.urdf')
        self.env = SpiderBotSimulator(self.spider_urdf_path, real_time_enabled=True, gui=True)
    
    def run(self) -> None:
        ic(self.mode)
        self.modes[self.mode]()
        
    def preprocess(self, observation: np.ndarray) -> np.ndarray:
        pos, vel = np.split(observation, [12])
        normal_pos = pos / (2 * np.pi)
        normal_vel = vel / self.env.spider.nominal_joint_velocity
        return np.array([*normal_pos, *normal_vel])
        
    def episode(self, agent: Agent, logging=False) -> None:
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
        
    def calc_fitness(agent: Agent) -> float:
        return 0
        
    def log_state(self, observation: np.ndarray, controls: np.ndarray) -> None:
        pass
        
    def train(self) -> None:
        # use class from spider_bot/training.py
        pass
    
    def save_model(self) -> None:
        pass
    
    def load_model(self) -> None:
        pass