from __future__ import print_function

import random
from typing import List
import heapq

import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
rdBase.DisableLog('rdApp.error')

import main.graph_ga.crossover as co, main.graph_ga.mutate as mu
from main.optimizer import BaseOptimizer


MINIMUM = 1e-10

def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # scores -> probs 
    population_scores = [s + MINIMUM for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
    return mating_pool


def reproduce(mating_pool, mutation_rate):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child


class GB_GA_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "graph_ga"
        self.starting_population = None
        self.carryover_smiles_pool = set()
        self.gp_train_smiles_set = set()

    def initialize_ga(self, config):
        if self.smi_file is not None:
            # Exploitation run
            self.starting_population = self.all_smiles[:config["population_size"]]
        else:
            # Exploration run
            self.starting_population = np.random.choice(self.all_smiles, config["population_size"])

    def make_new_population(self, population_mol, population_scores, config, pool):
        # new_population
        mating_pool = make_mating_pool(population_mol, population_scores, config["population_size"])
        offspring_mol = pool(delayed(reproduce)(mating_pool, config["mutation_rate"]) for _ in range(config["offspring_size"]))

        # add new_population
        population_mol += offspring_mol
        return self.sanitize(population_mol)


    def _optimize(self, oracle, config, inner_loop = False, max_generations = None, scoring_function = None):
        if (self.starting_population == None):
            self.initialize_ga(config)

        self.oracle.assign_evaluator(oracle)

        if inner_loop:
            return self.inner_optimize(oracle, config, scoring_function)
        else:
            self.outer_optimize(oracle, config, max_generations)

    def inner_optimize(self, oracle, config, scoring_function):
        start_cache = dict(scoring_function.cache)
        start_cache_size = len(start_cache)

        pool = joblib.Parallel(n_jobs=self.n_jobs)

        # select initial population
        top_smiles_at_bo_iter_start = [
                        s
                        for _, s in heapq.nlargest(
                            config["ga_pool_num_best"],
                            [
                                (self.oracle(smiles), smiles)
                                for smiles in self.oracle.mol_buffer.keys()
                            ],
                        )
            ]
        starting_population = set(top_smiles_at_bo_iter_start)
        starting_population.update(self.carryover_smiles_pool)
        if len(starting_population) < config["max_ga_start_population_size"]:
            samples_from_pool = random.sample(
                    self.oracle.mol_buffer.keys(),
                    min(len(self.oracle.mol_buffer.keys()), config["max_ga_start_population_size"]),
                )
            
            for s in samples_from_pool:
                    starting_population.add(s)
                    if (
                        len(starting_population)
                        >= config["max_ga_start_population_size"]
                    ):
                        break

        population_mol = [Chem.MolFromSmiles(s) for s in starting_population]
        population_scores = scoring_function(starting_population, batch = True)

        patience = 0
        generation = 0

        while True:
            if generation > config["max_generations"]:
                break
            population_mol = self.make_new_population(population_mol, population_scores, config, pool)

            population_smiles = [Chem.MolToSmiles(mol) for mol in population_mol]
            population_scores = scoring_function(population_smiles, batch=True)

            argsort = np.argsort(-np.asarray(population_scores))[:config["population_size"]]
            population_smiles = [population_smiles[i] for i in argsort]
            population_scores = [population_scores[i] for i in argsort]
            generation += 1

            sm_ac_list = list(scoring_function.cache.items())
            sm_ac_list.sort(reverse=True, key=lambda t: t[1])
            smiles_out = [s for s, v in sm_ac_list]
            acq_out = [v for s, v in sm_ac_list]
            self.starting_population = smiles_out

            
            sm_ac_list = list(scoring_function.cache.items())
            sm_ac_list.sort(reverse=True, key=lambda t: t[1])
            smiles_out = [s for s, v in sm_ac_list]
            acq_out = [v for s, v in sm_ac_list]

            # Add SMILES with high acquisition function values to the priority pool,
            # Since maybe they will have high acquisition function values next time
            self.carryover_smiles_pool = set()
            for s in smiles_out:
                if (
                    len(self.carryover_smiles_pool) < config["ga_pool_num_carryover"]
                    and s not in self.gp_train_smiles_set
                ):
                    self.carryover_smiles_pool.add(s)
                else:
                    break

            return smiles_out, acq_out
    
    def outer_optimize(self, oracle, config, max_generations):
        pool = joblib.Parallel(n_jobs=self.n_jobs)
        population_mol = [Chem.MolFromSmiles(s) for s in self.starting_population]
        population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])

        patience = 0
        generation = 0

        while True:
            if len(self.oracle) > 100:
                    self.sort_buffer()
                    old_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
            else:
                old_score = 0

            population_mol = self.make_new_population(population_mol, population_scores, config, pool)
            population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])

            # early stopping
            if len(self.oracle) > 100:
                self.sort_buffer()
                new_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
                # import ipdb; ipdb.set_trace()
                if (new_score - old_score) < 1e-3:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

                old_score = new_score
                    
            if self.finish:
                break
