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

class Driver:
    def __init__(self) -> None:
        self.modes = {
            'test': lambda: self.episode(Agent(24, 12), eval = True),
            'train': lambda: self.train()
        }
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', choices=self.modes.keys(), help='select mode from {train, test}', default='test')
        args = parser.parse_args()
        self.mode = args.mode
        
        self.cwd = os.getcwd()
        self.paths = {
            'models' :     os.path.join(self.cwd, 'models'),
            'figures':     os.path.join(self.cwd, 'figures')
            'spider-urdf': os.path.join(self.cwd, 'urdfs', 'spider_bot_v0.urdf')
        }

        self.env = SpiderBotSimulator(self.paths['spider-urdf'], real_time_enabled=True, gui=True)
    
    def run(self) -> None:
        ic(self.mode)
        self.modes[self.mode]()
        
    def preprocess(self, observation):
        pos, vel = np.split(observation, [12])
        normal_pos = pos / (2 * np.pi)
        normal_vel = vel / self.env.spider.nominal_joint_velocity
        return np.array([*normal_pos, *normal_vel])
        
    def episode(self, agent, logging=False, eval=False) -> None:
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
                    joint_pos.append([observation[0]])
                    joint_vel.append([observation[1]])
                    body_pos.append(info['pos'])
                
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

    
        #     joint_positions.append(pos)
        #     joint_velocities.append(vel)
        #     body_positions.append(info['pos'])

        #     i += 1
        #     if i % period == 0:
        #         alt = int(not alt)

        # self.graph_data(np.array(joint_positions).T, 
        #                 np.array(joint_velocities).T,
        #                 np.array(body_positions).T)
        
    def calc_fitness(agent):
        return 0
        
    def log_state(self, observation, controls):
        pass
    
    def graph_data(self, 
                   joint_positions:  np.array, 
                   joint_velocities: np.array,
                   body_positions:   np.array
                   ) -> None:

        print("positions:", joint_positions.shape)
        print("velocities:", joint_velocities.shape)
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
        plt.savefig(os.path.join(self.paths['figures'], 'body_position'))
        plt.show()
        #ax.grid()
        
        #print("Body:", body_positions)
        #pass
        
    def train(self):
        pass
    
    def save_model(self):
        pass
    
    def load_model(self):
        pass