"""

"""

import random
import numpy as np
from typing import Callable, Optional, Union, List, Any, Dict

# type aliases
MemberId = int
Generation = int

def make_dynamic_param(param: Any) -> Callable[[Optional[List[Any]], Optional[Dict[Any, Any]]], Any]:
    return lambda *args, **kwargs: param if not callable(param) else param

class Member:
    # base class
    def __init__(self):
        self.fitness = np.NINF
        self.id = None

class Genetics:

    def __init__(self,
                initial_population: List[Member],
                gens: int,
                evaluate_population: Callable[[Dict[MemberId, Member]], None],
                crossover_parents: Callable[[List[Member]], Member],
                mutate_member: Callable[[Member], Member],
                generation_cb: Optional[Callable[[List[Member]], None]],
                crossover_rate: Optional[Union[float, Callable[[Generation], float]]] = 0.3,
                mutation_rate: Optional[Union[float, Callable[[Generation], float]]] = 0.7,
                parent_competition_size: Optional[int] = 3,
                num_parents_per_child: Optional[int] = 2,
                mutation_stength_distribution: Optional[List[int]] = tuple(85 * [1] + 10 * [2] + 5 * [3]),
                ) -> None:
        
        self.population = initial_population
        for member in self.population:
            assert isinstance(member, Member)  # ensures all members enherit from Member
        self.id_to_member = {member.id: member for member in self.population}
        self.next_available_id = max(self.id_to_member.keys()) + 1
        self.population_size = len(initial_population)
        self.best_member = None
        
        self.target_gens = gens
        
        self.evaluate_population = evaluate_population
        self.crossover = crossover_parents
        self.mutate_member = mutate_member
        self.generation_callback = make_dynamic_param(generation_cb)
        
        self.crossover_rate = make_dynamic_param(crossover_rate)
        self.mutation_rate = make_dynamic_param(mutation_rate)
            
        self.parent_competition_size = make_dynamic_param(parent_competition_size)
        self.num_parents_per_child = make_dynamic_param(num_parents_per_child)
        self.mutation_strength_dist = make_dynamic_param(mutation_stength_distribution)
        
        self.gen = 0
		
    def evolve(self) -> None:
        while not self.is_done():
            self.gen += 1

            parents = self.select_parents()
            children = self.make_children({parent.id: parent for parent in parents})
            mutants = self.make_mutants()
            
            self.population += children + mutants

            self.evaluate_population(self.population)
            self.population.sort(key = lambda member: member.fitness, reverse=True)
            self.population = self.population[:self.population_size]
            
            local_best_member = self.population[0]
            if local_best_member.fitness > self.best_member.fitness:
                self.best_member = local_best_member
            
            assert len(self.id_to_member) == len(self.population) == self.population_size  # ids must be unique
            self.id_to_member = {member.id: member for member in self.population}
            self.generation_callback(self.population)
		
    def is_done(self):
        return self.gen > self.target_gens
  
    def select_parents(self) -> List[Member]:
        parents = []
        available_ids = list(self.id_to_member.keys())
        crossover_rate = self.crossover_rate(self.gen)
        competition_size = self.parent_competition_size(self.gen)
        parents_to_select = int(self.population_size * crossover_rate)
        for _ in range(parents_to_select):
            competitor_ids = random.sample(available_ids, competition_size)
            best_id = max(competitor_ids, key = lambda id: self.id_to_member[id].fitness)
            parents.append(self.id_to_member[best_id])
            available_ids.remove(best_id)
            
        return parents

    def make_children(self, parent_pool) -> List[Member]:
        children = []
        available_ids = list(parent_pool.keys())
        crossover_rate = self.crossover_rate(self.gen)
        num_parents_per_child = self.num_parents_per_child(self.gen)
        children_to_make = int(self.population_size * crossover_rate)
        for _ in range(children_to_make):
            parent_ids = random.sample(available_ids, num_parents_per_child)
            parents = [self.id_to_member[parent_id] for parent_id in parent_ids]
            for parent_id in parent_ids: 
                available_ids.remove(parent_id)
            children.append(self.crossover(parents))
            
        return children

    def make_mutants(self) -> List[Member]:
        mutants = []
        available_ids = list(self.id_to_member.keys())
        mutation_rate = self.mutation_rate(self.gen)
        mutation_strength_dist = self.mutation_strength_dist(self.gen)
        mutants_to_make = int(self.population_size * mutation_rate)
        for _ in range(mutants_to_make):
            mutant_id = random.choice(available_ids)
            mutant = self.id_to_member[mutant_id]
            num_mutations = random.choice(mutation_strength_dist)
            for _ in range(num_mutations):
                mutant = self.mutate(mutant)
            mutants.append(mutant)
            available_ids.remove(mutant_id)
            
        return mutants
    