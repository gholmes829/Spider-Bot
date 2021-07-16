"""

"""

import os
import pickle
import time
import numpy as np
from icecream import ic  # better printing for debugging
import matplotlib.pyplot as plt
import argparse

from types import FunctionType
from numpy.core.numeric import outer

from spider_bot.environments import SpiderBotSimulator
from spider_bot.agent import Agent
from spider_bot.training import Evolution
from graphing import *

class Driver:
    def __init__(self):
        self.modes = {
            'test': self.test,
            'train': self.train
        }
        
        parser = argparse.ArgumentParser()
        parser.add_argument('mode', choices = self.modes.keys(), help = 'determines which mode to run')
        args = parser.parse_args()
        self.mode = args.mode
        
        self.cwd = os.getcwd()
        self.paths = {
            'models' :     os.path.join(self.cwd, 'models'),
            'figures':     os.path.join(self.cwd, 'figures'),
            'spider-urdf': os.path.join(self.cwd, 'urdfs', 'spider_bot_v2.urdf')
        }

    def run(self) -> None:
        #ic(self.mode)
        self.modes[self.mode]()

    def make_env(self):
        return SpiderBotSimulator(self.paths['spider-urdf'], gui = False, fast_mode = True)

    def train(self) -> None:
        ev = Evolution(self.make_env, self.episode, gens=25)

        currentdir = os.getcwd()
        config_path = os.path.join(currentdir, 'neat/neat_config')

        before_time = time.time()
        winner_net = ev.run(config_path)
        time_to_train = time.time() - before_time
        print("Training successfully completed in " + str(time_to_train / 60.0) + " Minutes")

        self.save_model(winner_net, fn="neat_model")
        ev.close()

    def test(self):
        env =SpiderBotSimulator(self.paths['spider-urdf'])
        model = self.load_model("neat_model")
        agent = Agent(model, 30, 12)
        self.episode(agent, env, eval=True, verbose=True, max_steps=0)
        print('Done!')

    def episode(self, agent: Agent, env_var, terminate: bool = True, verbose: bool = False, max_steps: float = 5096, logging=False, eval=False) -> None:
        i = 0
        if callable(env_var):
            env = env_var()
        else:
            env = env_var
            
        done = False
        rewards = []
        observation = env.reset()
        controls = agent.predict(observation)
        joint_pos, joint_vel, joint_torques, body_pos = [], [], [], []

        try:
            while not terminate or (not done and (not max_steps or i < max_steps)):
                observation, reward, done, info = env.step(controls)
                rewards.append(reward)
                if logging:
                    self.log_state(observation, controls)
                if eval:
                    joint_pos.append(info['joint-pos'])
                    joint_vel.append(info['joint-vel'])
                    body_pos.append(info['body-pos'])
                    joint_torques.append(info['joint-torques'])
                
                controls = agent.predict(self.preprocess(observation, env))
                i += 1
        except KeyboardInterrupt:
            env.close()
        fitness = self.calc_fitness(rewards, i)
        if verbose:
            ic('Done!')
            ic(fitness)
        if eval:
            self.graph_data(
                np.array(joint_pos).T,
                np.array(joint_vel).T,
                np.array(body_pos).T,
                np.array(joint_torques).T
            )
            
        return fitness
    
    def preprocess(self, observation: np.ndarray, env) -> np.ndarray:
        joint_pos, joint_vel, orientation, vel = np.split(observation, [12, 24, 27])
        normal_joint_pos = joint_pos / env.spider.max_angle_range
        normal_joint_vel = joint_vel / env.spider.nominal_joint_velocity
        normal_orientation = orientation / (2 * np.pi)
        normal_vel = vel / env.max_spider_vel
        return np.array([*normal_joint_pos, *normal_joint_vel, *normal_orientation, *normal_vel])

    def log_state(self, observation: np.ndarray, controls: np.ndarray) -> None:
        pass

    @staticmethod
    def calc_fitness(rewards: list, steps: int) -> float:
        return sum(rewards)

    def save_model(self, model, fn: str = "model") -> None:
        with open(f'neat/{fn}.pickle', 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
            print("Successfully pickled winner net")
    
    def load_model(self, fn: str = "model"):
        with open(f'neat/{fn}.pickle', 'rb') as f:
            winner_net = pickle.load(f)
        return winner_net
    
    def graph_data(self, 
                    joint_positions:  np.array, 
                    joint_velocities: np.array,
                    body_positions:   np.array,
                    joint_torques:    np.array,
                    display_graphs:   bool = False
                    ) -> None:
        plt.style.use(["dark_background"])
        plt.rcParams.update({'font.size': 6})

        ax = GraphJointData(self.reorder_joints(joint_torques), "Joint Torques")
        plt.savefig(os.path.join(self.paths['figures'], 'joint_torques'), bbox_inches="tight", pad_inches = 0.25, dpi = 150)
        if display_graphs: plt.show()

        ax = GraphJointData(self.reorder_joints(joint_positions), "Joint Angles")#, ymin = -np.pi / 2, ymax = np.pi / 2)
        plt.savefig(os.path.join(self.paths['figures'], 'joint_angles'), bbox_inches="tight", pad_inches = 0.25, dpi = 150)
        if display_graphs: plt.show()

        ax = GraphJointData(self.reorder_joints(joint_velocities), "Joint Velocities")
        plt.savefig(os.path.join(self.paths['figures'], 'joint_velocities'), bbox_inches="tight", pad_inches = 0.25, dpi = 150)
        if display_graphs: plt.show()

        ax = GraphBodyTrajectory(body_positions)
        plt.savefig(os.path.join(self.paths['figures'], 'body_position'))
        if display_graphs: plt.show()

    def reorder_joints(self, joint_array: np.array) -> np.array:
        """ 
        Reformats an array of joint information as follows:
        0-3: Inner joints 
        4-7: Middle joints
        8-11: Outer joints
        Orange -> Green -> Yellow -> Purple
        Helps with graphing

        """
        outer_joints = joint_array[[0, 3, 6, 9]]
        middle_joints = joint_array[[1, 4, 7, 10]]
        inner_joints = joint_array[[2, 5, 8, 11]]
        return np.vstack((inner_joints, middle_joints, outer_joints))
