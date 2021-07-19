"""

"""

import numpy as np
import os
from tqdm import tqdm
from icecream import ic
import multiprocessing as mp
import neat
from functools import reduce
import psutil
from time import sleep

from spider_bot.agent import Agent

class Evolution:
    def __init__(self, make_env, fitness_function, gens = 100) -> None:
        self.make_env = make_env
        self.fitness_function = fitness_function
        self.generations = gens
        #self.pool = Pool(maxtasksperchild=1)
        self.progress = 0
        self.parallelize = False  # initialize to false, can change in <Evolution.run>

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
            ic('Making agents...')
            num_cores = psutil.cpu_count(logical=False)
            #assert num_cores == len(self.envs)
            
            agents = []
            for genome_id, genome in genomes:
                genome.fitness = 0
                agents.append(Agent(neat.nn.FeedForwardNetwork.create(genome, config), 30, 12, id=genome_id))
            ic(f'Num agents: {len(agents)}')
            batches = np.array_split(agents, num_cores)
            batch_sizes = [len(batch) for batch in batches]
            ic(f'{num_cores} batches: {batch_sizes}')
            
            # get workers
            progress_queue = mp.Queue()
            queues = [mp.Queue() for _ in range(num_cores)]
            workers = [mp.Process(target=self.eval_genome_batch, args=(batch, self.make_env, self.fitness_function, queue, progress_queue)) for i, (queue, batch) in enumerate(zip(queues, batches))]
            ic('Starting workers...')
            for worker in workers: worker.start()
            sleep(0.1)
            print(f'Located {len(psutil.Process().children())} out of {num_cores} processes...\n', flush=True)
            sleep(0.1)
            #while len(psutil.Process().children()) < num_cores:
            #    ic(len(psutil.Process().children()))
            #    sleep(0.1)
            #sleep(1)
            for i in tqdm(range(len(agents)), ascii=True):
                progress_queue.get()
            ic('Joining workers...')
            for worker in workers: worker.join(timeout=2.5)  # timeout shouldn't be needed in theory, but seems necessary to force physics clients closed
            
            ic('Getting results...')
            results = [[queue.get() for _ in range(batch_size)] for queue, batch_size in zip(queues, batch_sizes)]
            ic(f'{len(results)} batches of results: {[len(result) for result in results]}')
            results_flat = reduce(lambda base, next: base + next, results)
            #ic(f'Total num of results: {len(results_flat)}')
            for i, (genome_id, genome) in enumerate(genomes):
                fitness, agent_id = results_flat[i]
                assert agent_id == genome_id, f'Agent id, {agent_id}, and genome id, {genome_id}, don\'t match'
                genome.fitness = fitness
            ic(f'NEAT is evolving or something...')
            
        else:
            env = self.make_env()
            for genome_id, genome in tqdm(genomes, ascii=True):  # TODO: call eval_genome_batch with one batch -- entire thing
                genome.fitness = 0
                agent = Agent(neat.nn.FeedForwardNetwork.create(genome, config), 30, 12, id=genome_id)
                fitness, agent_id = self.fitness_function(agent, env, self.fitness_function)
                genome.fitness = fitness
    
    @staticmethod
    def eval_genome_batch(batch, make_env, fitness_func, queue, progress_queue):
        ic(os.getpid(), len(batch))
        env = make_env()
        for agent in batch:
            queue.put(fitness_func(agent, env))
            progress_queue.put(1)
        env.close()
        while env.physics_client.isConnected():  # sleep until client disconnects
            sleep(1)
    
    #def close(self):
    #    self.pool.close()
        
def eval_batch(agents, make_env, fitness_func):
    return [fitness_func(agent, make_env) for agent in agents]
    