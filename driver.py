"""
"""

import os
import pickle
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
import argparse

from spider_bot.utils import LivePlotter
from spider_bot.environments import SpiderBotSimulator
from spider_bot.agent import Agent
from spider_bot.training import Evolution
import graphing
from graphing import *
from spider_bot import settings

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

    def make_env(self, **kwargs):
        # change GUI to False here to use direct mode when training!!!
        return SpiderBotSimulator(self.paths['spider-urdf'], **kwargs)

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
        
        graph = LivePlotter(graphing.live_training_cb, graphing.initialize_axes, interval=1000)
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
        agent = Agent(model, settings.input_shape, 12)
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
        nn_output = [controls]

        try:
            while not terminate or (not done and (not max_steps or i < max_steps)):
                observation, reward, done, info = env.step(controls)
                rewards.append(reward)
                controls = agent.predict(self.preprocess(observation, env))
                i += 1

                if eval:
                    joint_pos.append(info['joint-pos'])
                    joint_vel.append(info['joint-vel'])
                    body_pos.append(info['body-pos'])
                    joint_torques.append(info['joint-torques'])
                    contact_data.append([int(e) for e in info['contact-data']])
                    ankle_pos.append(info['ankle-pos'])#[:][:][2])
                    vel = env.velocity
                    body_velocity.append(vel)
                    nn_output.append(controls)

        except KeyboardInterrupt:
            env.close()

        fitness = self.calc_fitness(env.initial_position, env.spider.get_pos(), env.steps, i, verbose=verbose)
        #fitness = self.calc_alt_fitness(env.initial_position, env.spider.get_pos(), env.steps, rewards, verbose=eval)

        if verbose:
            ic('Done!')
            ic(fitness)
            ic(f'Survived for {i} steps')
            ic(env.spider.get_pos(), env.initial_position)
            ic(np.sum(env.rising_edges, axis=1))
            ic(env.steps)

        if eval:
            self.graph_eval_data(
                np.array(joint_pos).T,
                np.array(joint_vel).T,
                np.array(body_pos).T,
                np.array(joint_torques).T,
                np.array(contact_data, dtype=int).T,
                np.array(ankle_pos).T,
                np.array(nn_output).T
            )

        return fitness, agent.id
    
    def preprocess(self, observation: np.ndarray, env) -> np.ndarray:
        joint_pos, joint_vel, orientation, sins = np.split(observation, [12, 24, 27])
        normal_joint_pos = joint_pos / env.spider.max_angle_range
        normal_joint_vel = joint_vel / env.spider.nominal_joint_velocity
        normal_orientation = orientation / (2 * np.pi)
        return np.array([*normal_joint_pos, *normal_joint_vel, *normal_orientation, *sins])

    @staticmethod
    def calc_fitness(initial_pos: np.array, current_pos: np.array, steps: np.array, T: int, verbose=False) -> float:
        """
        Uncomment measurements you wanna use!
        """
        dist_traveled = np.linalg.norm((current_pos - initial_pos)[:2])
        total_steps = np.sum(steps)
        #avg_outward_vel = dist_traveled / T
        #avg_steps_per_leg = np.mean(num_steps)
        steps_std = np.std(steps)
        #steps_cv = steps_std / avg_steps_per_leg
        #avg_steps_per_time = total_steps / T

        #default_min = 50
        #clipped_T = min(default_min + total_steps * 10, T)
        #fitness = 1e-3 * clipped_T ** 2  # float(np.sum(steps) ** 2 + 0.1 * min(T, 100))
        
        fitness = (1 + min(steps) ** 2) * (dist_traveled ** 2) * (1 + total_steps / T) / (1 + steps_std ** 2)
        if verbose:
            ic(steps)
            ic(total_steps)
            #ic(clipped_T)
            #ic(fitness)
        
        return float(fitness)

    @staticmethod
    def calc_alt_fitness(initial_pos: np.array, current_pos: np.array, steps: np.array, rewards: list, verbose=False) -> float:
        dist_traveled = current_pos[1] - initial_pos[1] #np.linalg.norm((current_pos - initial_pos)[:2])
        total_steps = np.sum(steps)
        total_reward = sum(rewards)
        fitness = total_reward + (100 * dist_traveled * total_steps ** 2)
        if verbose:
            ic(dist_traveled, steps, total_reward, fitness)
        return fitness

        
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
                    nn_output:        np.array,
                    display_graphs:   bool = False
                    ) -> None:

        """
        Uses data from a single test episode to create and save the following graphs:

        'joint_torques':    each joint's torque over time
        'joint_position':   each joint's position/angle over time
        'joint_velocities': each joint's velocity over time
        'nn_output':        the output in [0, 1] for each joint over time
        'contact_data':     binary data indicating whether a leg is touching the ground at a given time step
        'ankle_heights':    The z-position of each of the robot's ankles over time
        'body_position':    A 3D graph of the center of mass of the robot's body over time

        """

        plt.style.use(["dark_background"])
        plt.rcParams.update({'font.size': 6})

        ax = GraphJointData(self.reorder_joints(joint_torques), "Joint Torques")
        plt.savefig(os.path.join(self.paths['figures'], 'joint_torques'), bbox_inches="tight", pad_inches = 0.25, dpi = 150)
        if display_graphs: plt.show()

        ax = GraphJointData(self.reorder_joints(joint_positions), "Joint Angles")
        plt.savefig(os.path.join(self.paths['figures'], 'joint_angles'), bbox_inches="tight", pad_inches = 0.25, dpi = 150)
        if display_graphs: plt.show()

        ax = GraphJointData(self.reorder_joints(joint_velocities), "Joint Velocities")
        plt.savefig(os.path.join(self.paths['figures'], 'joint_velocities'), bbox_inches="tight", pad_inches = 0.25, dpi = 150)
        if display_graphs: plt.show()

        ax = GraphJointData(self.reorder_joints(nn_output), "Neural Network Output")
        plt.savefig(os.path.join(self.paths['figures'], 'nn_output'), bbox_inches="tight", pad_inches = 0.25, dpi = 150)
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