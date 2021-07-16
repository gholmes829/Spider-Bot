"""

"""

import numpy as np
import os
from tqdm import tqdm
from icecream import ic
from multiprocessing import Pool
import neat
from functools import reduce

from spider_bot.agent import Agent

class Evolution:
    def __init__(self, make_env, fitness_function, gens = 100) -> None:
        self.make_env = make_env
        self.fitness_function = fitness_function
        self.generations = gens
        self.pool = Pool(maxtasksperchild=1)
        self.progress = 0
        self.parallelize = False

    def run(self, config_file, parallelize = True):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)

        print("Creating Population...")
        p = neat.Population(config)

        p.add_reporter(neat.StdOutReporter(False))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        self.parallelize = parallelize

        winner = p.run(self.eval_genomes, self.generations)
        print('\nBest genome:\n{!s}'.format(winner.fitness))

        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        return ic(winner_net)

    def eval_genomes(self, genomes, config):
        # ToDo: get parallelization to work
        if self.parallelize:
            num_cores = os.cpu_count()
            #assert num_cores == len(self.envs)
            
            agents = []
            for genome_id, genome in genomes:
                genome.fitness = 0
                agents.append(Agent(neat.nn.FeedForwardNetwork.create(genome, config), 30, 12))
            
            batches = np.array_split(agents, num_cores)
            
            fitnesses = reduce(lambda base, next: base + next, self.pool.starmap(eval_batch, [(batch, self.make_env, self.fitness_function) for i, batch in enumerate(batches)]))
            for i, (genome_id, genome) in enumerate(genomes):
                genome.fitness = fitnesses[i]
            
            #jobs = [zip(agents[i:i + num_cores], self.envs) for i in range(0, len(agents), num_cores)]
            #for i, job in enumerate(num_cores):
            #    fitnesses = self.pool.starmap(self.fitness_function, job)
            #    for j in range(6):
            #        genomes[i * num_cores + j][1].fitness = fitnesses[j]
            
        else:
            env = self.make_env()
            for genome_id, genome in tqdm(genomes, ascii=True):
                genome.fitness = 0
                agent = Agent(neat.nn.FeedForwardNetwork.create(genome, config), 30, 12)
                genome.fitness = self.fitness_function(agent, env, self.fitness_function)
    
    def close(self):
        self.pool.close()
        
def eval_batch(agents, make_env, fitness_func):
    return [fitness_func(agent, make_env) for agent in agents]
    