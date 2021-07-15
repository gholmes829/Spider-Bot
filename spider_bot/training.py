"""

"""

import numpy as np
from icecream import ic
import neat

class Evolution:
    def __init__(self, env, gens = 100) -> None:
        self.generations = gens
        self.env = env

    def run(self, config_file):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)

        print("Creating Population...")
        p = neat.Population(config)

        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(neat.StatisticsReporter())

        winner = p.run(self.eval_genomes, self.generations)
        print('\nBest genome:\n{!s}'.format(winner.fitness))

        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        return ic(winner_net)

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            print("Genome#", genome_id, " fitness: ", end=" ")
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            genome.fitness += self.fitness_function(net)
            print(genome.fitness)

    def fitness_function(self, net):
        self.env.reset()
        observation = self.env.get_observation()
        for i in range(int(1e4)):
            observation, reward, done, info = self.env.step(net.activate(observation))
            if done:
                break
        return reward 