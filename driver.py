"""

"""

import os
import pickle
import numpy as np
from icecream import ic  # better printing for debugging
import matplotlib.pyplot as plt
import argparse

from spider_bot.utils import LivePlotter
from spider_bot.environments import SpiderBotSimulator
from spider_bot.agent import Agent
from spider_bot.training import Evolution
import graphing
from graphing import *

class Driver:
    def __init__(self):
        self.modes = {
            'test': self.test_bot,
            'train': self.train
        }
        
        args = self.parse_args()
        self.mode = args.mode
        
        self.cwd = os.getcwd()
        self.paths = {
            'models' :     os.path.join(self.cwd, 'models'),
            'figures':     os.path.join(self.cwd, 'figures'),
            'spider-urdf': os.path.join(self.cwd, 'urdfs', 'spider_bot_v2.urdf'),
            'checkpoints': os.path.join(self.cwd, 'checkpoints')
        }
        self.model_name = None
        self.fitnesses = []

    def run(self) -> None:
        self.modes[self.mode]()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('mode', choices = self.modes.keys(), help = 'determines which mode to run')
        args = parser.parse_args()
        return args

    def make_env(self, gui = False, fast_mode=True, verbose=False):
        # change GUI to false here to use direct mode when training!!!
        return SpiderBotSimulator(self.paths['spider-urdf'], gui = gui, fast_mode = fast_mode)

    def get_model_name(self):
        valid_model_name = False 
        while not valid_model_name:
            model_name = input("Name for this model: ") + '.pickle'
            if os.path.isfile(os.path.join(self.paths['models'], model_name)):
                print("That name is already being used. ")
            else:
                valid_model_name = True
                self.paths['session'] = os.path.join(self.cwd, model_name)
                self.model_name = model_name

    def train(self) -> None:
        self.get_model_name()
        gens = int(input("Number of generations: "))
        
        checkpoint_dir = os.path.join(self.paths['checkpoints'], self.model_name[:-7])
        os.mkdir(checkpoint_dir) # create a directory to save checkpoints
        
        graph = LivePlotter(graphing.live_training_cb, graphing.make_training_fig)
        graph.start()
        ev = Evolution(self.make_env, self.episode, checkpoint_dir, graph, gens=gens)

        config_path = os.path.join(self.cwd, 'neat/neat_config')

        (winner_net, fitnesses), time_to_train = ev.run(config_path)

        print("Training successfully completed in " + str(time_to_train / 60.0) + " Minutes")

        self.graph_training_data(np.array(fitnesses))
        self.save_model(winner_net)
        
        print('Close graph to end training...')
        graph.close()

    def test_bot(self):
        model = self.load_model()
        if model is None:
            return
        env = SpiderBotSimulator(self.paths['spider-urdf'])
        agent = Agent(model, 30, 12)
        self.episode(agent, env, eval=True, verbose=True, max_steps=0)
        print('Done!')

    def episode(self, agent: Agent, env_var, terminate: bool = True, verbose: bool = False, max_steps: float = 2048, eval=False) -> None:
        i = 0
        if callable(env_var):
            env = env_var()
        else:
            env = env_var
            
        done = False
        rewards = []
        observation = env.reset()
        controls = agent.predict(self.preprocess(observation, env))
        joint_pos, joint_vel, joint_torques, body_pos, contact_data, ankle_pos = [], [], [], [], [], []
        body_velocity = []

        try:
            while not terminate or (not done and (not max_steps or i < max_steps)):
                observation, reward, done, info = env.step(controls)
                rewards.append(reward)
                
                if eval:
                    joint_pos.append(info['joint-pos'])
                    joint_vel.append(info['joint-vel'])
                    body_pos.append(info['body-pos'])
                    joint_torques.append(info['joint-torques'])
                    contact_data.append([int(e) for e in info['contact-data']])
                    ankle_pos.append(info['ankle-pos'])#[:][:][2])
                    vel = env.velocity
                    body_velocity.append(vel)
                
                controls = agent.predict(self.preprocess(observation, env))
                i += 1
                
        except KeyboardInterrupt:
            env.close()
        filtered_rising_edges = env.get_filtered_rising_edges()
        ic(env.steps)
        fitness = sum(env.steps)#self.calc_fitness(env.spider.get_pos(), env.initial_position, filtered_rising_edges)

        if verbose:
            ic('Done!')
            ic(fitness)
            ic(f'Survived for {i} steps')
            ic(np.sum(env.rising_edges, axis=1))
            ic(np.sum(filtered_rising_edges, axis=1))
 
        if eval:
            self.graph_eval_data(
                np.array(joint_pos).T,
                np.array(joint_vel).T,
                np.array(body_pos).T,
                np.array(joint_torques).T,
                np.array(contact_data, dtype=int).T,
                np.array(ankle_pos).T
            )
            ic(np.sum(body_velocity, axis=0))

        return fitness, agent.id
    
    def preprocess(self, observation: np.ndarray, env) -> np.ndarray:
        joint_pos, joint_vel, orientation, vel = np.split(observation, [12, 24, 27])
        normal_joint_pos = joint_pos / env.spider.max_angle_range
        normal_joint_vel = joint_vel / env.spider.nominal_joint_velocity
        normal_orientation = orientation / (2 * np.pi)
        normal_vel = vel / env.max_spider_vel
        return np.array([*normal_joint_pos, *normal_joint_vel, *normal_orientation, *normal_vel])

    @staticmethod
    def calc_fitness(current_pos: np.array, initial_pos: np.array, filtered_rising_edges: np.array) -> float:
        """
        """
        T = len(filtered_rising_edges[0])
        num_edges = [sum(leg) for leg in filtered_rising_edges]
        target_time_per_step = 150
        target_num_steps = T / target_time_per_step
        avg_edges_per_leg = np.mean(num_edges)
        modifier = 1
        if avg_edges_per_leg < target_num_steps:
            modifier = avg_edges_per_leg / target_num_steps
        return 10 * np.linalg.norm((current_pos - initial_pos)[:2]) * modifier + np.sqrt(min(100, T))
        
    def save_model(self, model) -> None:
        with open(os.path.join(self.paths['models'], self.model_name), 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
            print("Successfully pickled winner net")
    
    def load_model(self, fn: str = "model"):
        models = []
        for file in os.listdir(self.paths['models']):
            if file.endswith(".pickle"): models.append(file)
        num_models = len(models)
        if num_models == 0:
            print("No saved models. ")
        else:
            print()
            msg = "Select model to use:"
            for i, model in enumerate(models, start=1):
                msg += "\n    " + str(i) + ") " + str(model[:-7])
            cancelIndex = num_models + 1
            msg += "\n    " + str(cancelIndex) + ") Cancel"
            print(msg)
            try:
                index = int(input("Choice: ")) - 1
                print(type(index))
                if (index >= 0 and index < cancelIndex - 1):
                    with open(os.path.join(self.paths['models'], models[index]), 'rb') as f:
                        winner_net = pickle.load(f)
                    return winner_net
            except ValueError:
                pass
        return None
    
    def graph_training_data(self, fitnesses: np.array):
        plt.style.use(["dark_background"])
        ax = GraphFitness(fitnesses)
        plt.savefig(os.path.join(self.paths['figures'], 'fitness_over_time'), bbox_inches="tight", pad_inches = 0.25, dpi = 150)
        plt.show()

    def graph_eval_data(self, 
                    joint_positions:  np.array, 
                    joint_velocities: np.array,
                    body_positions:   np.array,
                    joint_torques:    np.array,
                    contact_data:     list,
                    ankle_pos:        np.array,
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

        ax = GraphContactData(contact_data)
        plt.savefig(os.path.join(self.paths['figures'], 'contact_data'), bbox_inches="tight", pad_inches = 0.25, dpi = 150)
        if display_graphs: plt.show()

        ax = GraphAnkleHeights(ankle_pos[2])
        plt.savefig(os.path.join(self.paths['figures'], 'ankle_heights'))
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
