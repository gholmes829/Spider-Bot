"""

"""

import os
import numpy as np
from icecream import ic  # better printing for debugging
import matplotlib.pyplot as plt
import argparse

from spider_bot.environments import SpiderBotSimulator
from spider_bot.agent import Agent
from spider_bot.training import Evolution
from graphing import *

class Driver:
    def __init__(self):
        self.modes = {
            'test': lambda: self.episode(Agent(24, 12), eval = True),
            'train': lambda: self.train()
        }
        
        parser = argparse.ArgumentParser()
        parser.add_argument('mode', choices = self.modes.keys(), help = 'determines which mode to run')
        args = parser.parse_args()
        self.mode = args.mode
        
        self.cwd = os.getcwd()
        self.paths = {
            'models' :     os.path.join(self.cwd, 'models'),
            'figures':     os.path.join(self.cwd, 'figures'),
            'spider-urdf': os.path.join(self.cwd, 'urdfs', 'spider_bot_v0.urdf')
        }

        self.env = SpiderBotSimulator(self.paths['spider-urdf'], real_time_enabled=True, gui=True)
    
    def run(self) -> None:
        ic(self.mode)
        self.modes[self.mode]()
        
    def preprocess(self, observation: np.ndarray) -> np.ndarray:
        pos, vel = np.split(observation, [12])
        normal_pos = pos / (2 * np.pi)
        normal_vel = vel / self.env.spider.nominal_joint_velocity
        return np.array([*normal_pos, *normal_vel])
        
    def episode(self, agent: Agent, logging=False, eval=False) -> None:
        i = 0
        max_steps = 5096
        done = False
        rewards = []
        observation = self.env.reset()
        controls = agent.predict(observation)
        joint_pos, joint_vel, body_pos = [], [], []
        
        try:
            while not done and i < max_steps:
                observation, reward, done, info = self.env.step(controls)
                rewards.append(reward)
                if logging:
                    self.log_state(observation, controls)
                if eval:
                    joint_pos.append(info['joint-pos'])
                    joint_vel.append(info['joint-vel'])
                    body_pos.append(info['body-pos'])
                
                controls = agent.predict(self.preprocess(observation))
                i += 1
        except KeyboardInterrupt:
            self.env.close()

        if eval:
            self.graph_data(
                np.array(joint_pos).T,
                np.array(joint_vel).T,
                np.array(body_pos).T
            )
            
        return rewards
        
    def calc_fitness(agent: Agent) -> float:
        return 0
        
    def log_state(self, observation: np.ndarray, controls: np.ndarray) -> None:
        pass
    
    def graph_data(self, 
                   joint_positions:  np.array, 
                   joint_velocities: np.array,
                   body_positions:   np.array,
                   display_graphs:   bool = True
                   ) -> None:

        print("positions:", joint_positions.shape)
        print("velocities:", joint_velocities.shape)
        plt.style.use(["dark_background"])

        ax = GraphJointVelocities(joint_velocities[self.env.spider.outer_joints], 'Outer')
        plt.savefig(os.path.join(self.paths['figures'], 'outer_joint_velocities'))
        if display_graphs: plt.show()

        ax = GraphJointVelocities(joint_velocities[self.env.spider.inner_joints], 'Inner')
        plt.savefig(os.path.join(self.paths['figures'], 'inner_joint_velocities'))
        if display_graphs: plt.show()

        ax = GraphBodyTrajectory(body_positions)
        plt.savefig(os.path.join(self.paths['figures'], 'body_position'))
        if display_graphs: plt.show()


        
    def train(self) -> None:
        # use class from spider_bot/training.py
        pass
    
    def save_model(self) -> None:
        pass
    
    def load_model(self) -> None:
        pass