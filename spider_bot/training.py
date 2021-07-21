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
import pickle

from spider_bot.agent import Agent
from tests.util import timed

class Evolution:
    def __init__(self, make_env, fitness_function, checkpoint_dir, gens = 100) -> None:
        self.make_env = make_env
        self.fitness_function = fitness_function
        self.generations = gens
        self.current_generation = 0
        self.num_workers = 1

        self.progress = 0
        self.average_fitnesses = []
        self.average_fitnesses = []
        self.checkpoint_dir = checkpoint_dir

    @timed
    def run(self, config_file, num_workers = -1):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)

        print("Creating Population...")
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(False))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        self.num_workers = num_workers if num_workers > 0 else psutil.cpu_count(logical=False)

        winner = p.run(self.eval_genomes, self.generations)
        print('\nBest genome:\n{!s}'.format(winner.fitness))

        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        return ic(winner_net, self.average_fitnesses)

    def eval_genomes(self, genomes, config):        
        for genome_id, genome in genomes: genome.fitness = 0
        agents = [Agent(neat.nn.FeedForwardNetwork.create(genome, config), 30, 12, id=genome_id) for genome_id, genome in genomes]
        agent_batches = np.array_split(agents, self.num_workers)
        
        progress_queue = mp.Queue()
        result_queues = [mp.Queue() for _ in range(self.num_workers)]
        workers = [mp.Process(target=self.eval_genome_batch, args=(agent_batch, self.make_env, self.fitness_function, result_queue, progress_queue)) for result_queue, agent_batch in zip(result_queues, agent_batches)]
        for worker in workers: worker.start()
        
        sleep(1)
        children_found = len(psutil.Process().children())
        if children_found != self.num_workers: print(f'WARNING: only located {children_found} out of {self.num_workers} processes...\n', flush=True)

        for _ in tqdm(range(len(agents)), ascii=True): progress_queue.get()  # loading bar
            
        results = [[queue.get() for _ in range(len(batch))] for queue, batch in zip(result_queues, agent_batches)]
        for queue in result_queues: queue.close()
        progress_queue.close()
            
        for worker in workers: worker.join(5)  # timeout shouldn't be needed in theory, but just in case physics clients get stuck
        exit_codes = [worker.exitcode for worker in workers]
        if any(exit_codes): print(f'WARNING: recieved non-zero exit code, {exit_codes}', flush=True)
            
        for worker in workers: worker.close()

        results_flat = reduce(lambda base, next: base + next, results)

        total_fitess = 0
        best = None
        
        for i, (genome_id, genome) in enumerate(genomes):
            fitness, agent_id = results_flat[i]
            assert agent_id == genome_id, f'Agent id, {agent_id}, and genome id, {genome_id}, don\'t match'
            genome.fitness = fitness
            total_fitess += fitness
            if best is None or genome.fitness > best.fitness: best = genome
            
        self.checkpoint(best, config)
        self.average_fitnesses.append(total_fitess / len(genomes))
            
        self.current_generation += 1
    
    @staticmethod
    def eval_genome_batch(batch, make_env, fitness_func, result_queue, progress_queue):
        print(f'New process with PID: {os.getpid()}, processing {len(batch)} agents', flush=True)
        env = make_env()
        for agent in batch:
            result_queue.put(fitness_func(agent, env))
            progress_queue.put(1)
        env.close()

        while env.physics_client.isConnected():  # sleep until client disconnects
            sleep(0.1)

    def checkpoint(self, best_genome, config):
        genome_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        filename = os.path.join(self.checkpoint_dir, "gen" + str(self.current_generation) + "-" + str(round(best_genome.fitness, 2))) + '.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(genome_net, f, pickle.HIGHEST_PROTOCOL)
            print("Saved model with fitness ", best_genome.fitness)