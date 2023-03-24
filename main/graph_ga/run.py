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

    def initialize_ga(self, config):
        if self.smi_file is not None:
            # Exploitation run
            self.starting_population = self.all_smiles[:config["population_size"]]
        else:
            # Exploration run
            self.starting_population = np.random.choice(self.all_smiles, config["population_size"])


    def _optimize(self, oracle, config, inner_loop = False, max_generations = None, scoring_function = None):

        if (self.starting_population == None):
            self.initialize_ga(config)

        if inner_loop:
            start_cache = dict(scoring_function.cache)
            start_cache_size = len(start_cache)
        else:
            self.oracle.assign_evaluator(oracle)

        pool = joblib.Parallel(n_jobs=self.n_jobs)

        starting_population = self.starting_population

        # select initial population
        # population_smiles = heapq.nlargest(config["population_size"], starting_population, key=oracle)
        population_smiles = starting_population
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        if inner_loop:
            population_scores = scoring_function(population_smiles, batch = True)
        else:
            population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])

        patience = 0
        generation = 0

        while True:
            if inner_loop:
                if generation > config["max_generations"]:
                    break
            else:
                if len(self.oracle) > 100:
                    self.sort_buffer()
                    old_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
                else:
                    old_score = 0

            # new_population
            mating_pool = make_mating_pool(population_mol, population_scores, config["population_size"])
            offspring_mol = pool(delayed(reproduce)(mating_pool, config["mutation_rate"]) for _ in range(config["offspring_size"]))

            # add new_population
            population_mol += offspring_mol
            population_mol = self.sanitize(population_mol)

            # stats
            old_scores = population_scores
            if inner_loop:
                population_scores = scoring_function(population_smiles, batch=True)
            else:
                population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]

            if inner_loop:
                # trim population
                argsort = np.argsort(-np.asarray(population_scores))[:population_size]
                population_smiles = [population_smiles[i] for i in argsort]
                population_scores = [population_scores[i] for i in argsort]
                
            ### early stopping
            if inner_loop:
                sm_ac_list = list(scoring_function.cache.items())
                sm_ac_list.sort(reverse=True, key=lambda t: t[1])
                smiles_out = [s for s, v in sm_ac_list]
                acq_out = [v for s, v in sm_ac_list]
                self.starting_population = smiles_out
            else:
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

        if inner_loop:
            # updating starting population for next round
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
            self.starting_population = set(top_smiles_at_bo_iter_start)
