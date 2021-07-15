"""

"""

import numpy as np
from tqdm import tqdm
from icecream import ic
import neat

from spider_bot.agent import Agent

class Evolution:
    def __init__(self, env, fitness_function, gens = 100) -> None:
        self.env = env
        self.fitness_function = fitness_function
        self.generations = gens

    def run(self, config_file):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)

        print("Creating Population...")
        p = neat.Population(config)

        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        winner = p.run(self.eval_genomes, self.generations)
        print('\nBest genome:\n{!s}'.format(winner.fitness))

        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        return ic(winner_net)

    def eval_genomes(self, genomes, config):
        for genome_id, genome in tqdm(genomes):
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = self.fitness_function(Agent(net, 24, 12))