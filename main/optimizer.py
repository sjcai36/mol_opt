import os
import yaml
import random
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import tdc
from tdc.generation import MolGen
import wandb
import gpytorch
import functools
from main.utils.chem import *
#from main.gpbo_general.gp import TanimotoGP, batch_predict_mu_var_numpy, fit_gp_hyperparameters, get_trained_gp
#from main.gpbo_general.fingerprints import smiles_to_fingerprint_stack, smiles_to_fp_array
#from main.gpbo_general.acquisition_funcs import acq_f_of_time
##from main.gpbo_general.function_utils import CachedFunction, CachedBatchFunction
#from main.utils.choose_optimizer import choose_optimizer


class Objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls


class Oracle:
    def __init__(self, args=None, mol_buffer={}):
        self.name = None
        self.evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
        self.mol_buffer = mol_buffer
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.last_log = 0

    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix=None):
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def log_intermediate(self, mols=None, scores=None, finish=False):

        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[:self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m) for m in mols]
                n_calls = len(self.mol_buffer)
        
        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        avg_sa = np.mean(self.sa_scorer(smis))
        diversity_top100 = self.diversity_evaluator(smis)
        
        print(f'{n_calls}/{self.max_oracle_calls} | '
                f'avg_top1: {avg_top1:.3f} | '
                f'avg_top10: {avg_top10:.3f} | '
                f'avg_top100: {avg_top100:.3f} | '
                f'avg_sa: {avg_sa:.3f} | '
                f'div: {diversity_top100:.3f}')

        # try:
        wandb.log({
            "avg_top1": avg_top1, 
            "avg_top10": avg_top10, 
            "avg_top100": avg_top100, 
            "auc_top1": top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls),
            "auc_top10": top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls),
            "auc_top100": top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls),
            "avg_sa": avg_sa,
            "diversity_top100": diversity_top100,
            "n_oracle": n_calls,
            # "best_mol": wandb.Image(Draw.MolsToGridImage([Chem.MolFromSmiles(item[0]) for item in temp_top10], 
            #           molsPerRow=5, subImgSize=(200,200), legends=[f"f = {item[1][0]:.3f}, #oracle = {item[1][1]}" for item in temp_top10]))
        })


    def __len__(self):
        return len(self.mol_buffer) 

    def score_smi(self, smi):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        else:
            smi = Chem.MolToSmiles(mol)
            if smi in self.mol_buffer:
                pass
            else:
                self.mol_buffer[smi] = [float(self.evaluator(smi)), len(self.mol_buffer)+1]
            return self.mol_buffer[smi][0]
    
    def __call__(self, smiles_lst):
        """
        Score
        """
        if type(smiles_lst) == list:
            score_list = []
            for smi in smiles_lst:
                score_list.append(self.score_smi(smi))
                if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                    self.sort_buffer()
                    self.log_intermediate()
                    self.last_log = len(self.mol_buffer)
                    self.save_result(self.task_label)
        else:  ### a string of SMILES 
            score_list = self.score_smi(smiles_lst)
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result(self.task_label)
        return score_list

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls


class BaseOptimizer:

    def __init__(self, args=None):
        self.model_name = "Default"
        self.args = args
        self.n_jobs = args.n_jobs
        # self.pool = joblib.Parallel(n_jobs=self.n_jobs)
        self.smi_file = args.smi_file
        self.oracle = Oracle(args=self.args)
        if self.smi_file is not None:
            self.all_smiles = self.load_smiles_from_file(self.smi_file)
        else:
            data = MolGen(name = 'ZINC')
            self.all_smiles = data.get_data()['smiles'].tolist()
            
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.filter = tdc.chem_utils.oracle.filter.MolFilter(filters = ['PAINS', 'SureChEMBL', 'Glaxo'], property_filters_flag = False)

    # def load_smiles_from_file(self, file_name):
    #     with open(file_name) as f:
    #         return self.pool(delayed(canonicalize)(s.strip()) for s in f)
            
    def sanitize(self, mol_list):
        new_mol_list = []
        smiles_set = set()
        for mol in mol_list:
            if mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles is not None and smiles not in smiles_set:
                        smiles_set.add(smiles)
                        new_mol_list.append(mol)
                except ValueError:
                    print('bad smiles')
        return new_mol_list
        
    def sort_buffer(self):
        self.oracle.sort_buffer()
    
    def log_intermediate(self, mols=None, scores=None, finish=False):
        self.oracle.log_intermediate(mols=mols, scores=scores, finish=finish)
    
    def log_result(self):

        print(f"Logging final results...")

        # import ipdb; ipdb.set_trace()

        log_num_oracles = [100, 500, 1000, 3000, 5000, 10000]
        assert len(self.mol_buffer) > 0 

        results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))
        if len(results) > 10000:
            results = results[:10000]
        
        results_all_level = []
        for n_o in log_num_oracles:
            results_all_level.append(sorted(results[:n_o], key=lambda kv: kv[1][0], reverse=True))
        
        # Currently logging the top-100 moelcules, will update to PDD selection later
        data = [[i+1, results_all_level[-1][i][1][0], results_all_level[-1][i][1][1], \
                wandb.Image(Draw.MolToImage(Chem.MolFromSmiles(results_all_level[-1][i][0]))), results_all_level[-1][i][0]] for i in range(100)]
        columns = ["Rank", "Score", "#Oracle", "Image", "SMILES"]
        wandb.log({"Top 100 Molecules": wandb.Table(data=data, columns=columns)})
        
        # Log batch metrics at various oracle calls
        data = [[log_num_oracles[i]] + self._analyze_results(r) for i, r in enumerate(results_all_level)]
        columns = ["#Oracle", "avg_top100", "avg_top10", "avg_top1", "Diversity", "avg_SA", "%Pass", "Top-1 Pass"]
        wandb.log({"Batch metrics at various level": wandb.Table(data=data, columns=columns)})
        
    def save_result(self, suffix=None):

        print(f"Saving molecules...")
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)
    
    def _analyze_results(self, results):
        results = results[:100]
        scores_dict = {item[0]: item[1][0] for item in results}
        smis = [item[0] for item in results]
        scores = [item[1][0] for item in results]
        smis_pass = self.filter(smis)
        if len(smis_pass) == 0:
            top1_pass = -1
        else:
            top1_pass = np.max([scores_dict[s] for s in smis_pass])
        return [np.mean(scores), 
                np.mean(scores[:10]), 
                np.max(scores), 
                self.diversity_evaluator(smis), 
                np.mean(self.sa_scorer(smis)), 
                float(len(smis_pass) / 100), 
                top1_pass]

    def reset(self):
        del self.oracle
        self.oracle = Oracle(args=self.args)

    @property
    def mol_buffer(self):
        return self.oracle.mol_buffer

    @property
    def finish(self):
        return self.oracle.finish
        
    def _optimize(self, oracle, config):
        raise NotImplementedError
            
    def hparam_tune(self, oracles, hparam_space, hparam_default, count=5, num_runs=3, project="tune"):
        seeds = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        seeds = seeds[:num_runs]
        hparam_space["name"] = hparam_space["name"]
        
        def _func():
            with wandb.init(config=hparam_default, allow_val_change=True) as run:
                avg_auc = 0
                for oracle in oracles:
                    auc_top10s = []
                    for seed in seeds:
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        random.seed(seed)
                        config = wandb.config
                        self._optimize(oracle, config)
                        auc_top10s.append(top_auc(self.oracle.mol_buffer, 10, True, self.oracle.freq_log, self.oracle.max_oracle_calls))
                        self.reset()
                    avg_auc += np.mean(auc_top10s)
                wandb.log({"avg_auc": avg_auc})
            
        sweep_id = wandb.sweep(hparam_space)
        # wandb.agent(sweep_id, function=_func, count=count, project=self.model_name + "_" + oracle.name)
        wandb.agent(sweep_id, function=_func, count=count, entity="sjcai")
        
    def optimize(self, oracle, config, seed=0, project="UROP"):

        run = wandb.init(project=project, config=config, reinit=True, entity="sjcai")
        wandb.run.name = self.model_name + "_" + oracle.name + "_" + wandb.run.id
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.seed = seed 
        self.oracle.task_label = self.model_name + "_" + oracle.name + "_" + str(seed)
        self._optimize(oracle, config)
        if self.args.log_results:
            self.log_result()
        self.save_result(self.model_name + "_" + oracle.name + "_" + str(seed))
        # self.reset()
        run.finish()
        self.reset()


    def production(self, oracle, config, num_runs=5, project="production"):
        seeds = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        # seeds = [23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        if num_runs > len(seeds):
            raise ValueError(f"Current implementation only allows at most {len(seeds)} runs.")
        seeds = seeds[:num_runs]
        for seed in seeds:
            self.optimize(oracle, config, seed, project)
            self.reset()
""" 
    def gpbo_optimize(self, oracle, config):
        # Functions to do retraining
        def get_inducing_indices(y):
            
            To reduce the training cost of GP model, we only select
            top-n_train_gp_best and n_train_gp_rand random samples from data.
            
            argsort = np.argsort(-y)  # Biggest first
            best_idxs = list(argsort[: config["n_train_gp_best"]])
            remaining_idxs = list(argsort[config["n_train_gp_best"] :])
            if len(remaining_idxs) <= config["n_train_gp_rand"]:
                rand_idxs = remaining_idxs
            else:
                rand_idxs = random.sample(remaining_idxs, k=config["n_train_gp_rand"])
            return sorted(best_idxs + rand_idxs)

        def refit_gp_change_subset(bo_iter, gp_model, bo_state_dict):
            gp_model.train()
            x = gp_model.train_inputs[0]
            y = gp_model.train_targets.detach().cpu().numpy()
            idxs = get_inducing_indices(y)
            gp_model.set_train_data(
                inputs=x[idxs].clone(),
                targets=gp_model.train_targets[idxs].clone(),
                strict=False,
            )
            gp_model.eval()

        self.oracle.assign_evaluator(oracle)
        torch.set_default_dtype(torch.float64)
        NP_DTYPE = np.float64

        # Load start smiles
        if self.smi_file is not None:
            # Exploitation run
            starting_population = self.all_smiles[: config["initial_population_size"]]
        else:
            # Exploration run
            starting_population = np.random.choice(
                self.all_smiles, config["initial_population_size"]
            )

        smiles_pool = set(starting_population)

        # Create data; fit exact GP

        gp_train_smiles = list(smiles_pool)
        values = self.oracle(gp_train_smiles)

        x_train, invalid_idx = smiles_to_fingerprint_stack(
            gp_train_smiles, config, NP_DTYPE
        )
        y_train = np.asarray(values).astype(NP_DTYPE)
        y_train = np.delete(y_train, invalid_idx)

        ind_idx_start = get_inducing_indices(y_train)
        x_train = torch.as_tensor(x_train)
        y_train = torch.as_tensor(y_train)
        gp_model = get_trained_gp(x_train[ind_idx_start], y_train[ind_idx_start])

        # Run GP-BO
        with gpytorch.settings.sgpr_diagonal_correction(False):
            # Set up which SMILES the GP should be trained on
            # If not given, it is assumed that the GP is trained on all known smiles
            start_cache = self.oracle.mol_buffer
            start_cache_size = len(self.oracle)
            if gp_train_smiles is None:
                #     "No GP training SMILES given. "
                #     f"Will default to training on the {start_cache_size} SMILES with known scores."
                gp_train_smiles_set = set(start_cache.keys())
            else:
                gp_train_smiles_set = set(gp_train_smiles)
            del gp_train_smiles  # should refer to new variables later on; don't want to use by mistake

            # Keep a pool of all SMILES encountered (used for seeding GA)
            if smiles_pool is None:
                smiles_pool = set()
            else:
                smiles_pool = set(smiles_pool)
            smiles_pool.update(start_cache.keys())
            smiles_pool.update(gp_train_smiles_set)
            assert (
                len(smiles_pool) > 0
            ), "No SMILES were provided to the algorithm as training data, known scores, or a SMILES pool."

            # Handle edge case of no training data
            if len(gp_train_smiles_set) == 0:
                #     f"No SMILES were provided to train GP. A random one will be chosen from the pool to start training."
                random_smiles = random.choice(list(smiles_pool))
                gp_train_smiles_set.add(random_smiles)
                del random_smiles

            # Evaluate scores of training data (ideally should all be known)
            num_train_data_not_known = len(
                gp_train_smiles_set - set(start_cache.keys())
            )
            gp_train_smiles_list = list(gp_train_smiles_set)
            gp_train_smiles_scores = self.oracle(list(gp_train_smiles_set))

            # Store GP training data
            y_train_np = np.array(gp_train_smiles_scores).astype(NP_DTYPE)

            x_train_np, invalid_idx = smiles_to_fingerprint_stack(
                gp_train_smiles_list, config, NP_DTYPE
            )
            y_train_np = np.delete(y_train_np, invalid_idx)

            print("Initial GP model training")
            gp_model.set_train_data(
                inputs=torch.as_tensor(x_train_np),
                targets=torch.as_tensor(y_train_np),
                strict=False,
            )

            # State variables for BO loop
            carryover_smiles_pool = set()
            bo_query_res = list()
            bo_state_dict = dict(
                gp_model=gp_model,
                gp_train_smiles_list=gp_train_smiles_list,
                bo_query_res=bo_query_res,
                scoring_function=self.oracle,
            )

            # Initial fitting of GP hyperparameters
            refit_gp_change_subset(
                bo_iter=0, gp_model=gp_model, bo_state_dict=bo_state_dict
            )

            # Actual BO loop
            for bo_iter in range(1, config["max_bo_iter"] + 1):

                if self.finish:
                    break

                # Current acquisition function
                curr_acq_func = acq_f_of_time(bo_iter, bo_state_dict)

                scoring_function = functools.partial(
                    self._acq_func_smiles,
                    gp_model=gp_model,
                    acq_func_np=curr_acq_func,
                    config=config,
                )

                scoring_function = CachedBatchFunction(scoring_function)

                # Optimize acquisition function
                print("Maximizing BO surrogate...")
                acq_smiles, acq_vals = self.internal_optimizer._optimize(
                    scoring_function=scoring_function,
                    config=config,
                    inner_loop=True,
                )

                # Now that new SMILES were generated, add them to the pool
                smiles_pool.update(acq_smiles)

                # Greedily choose SMILES to be in the BO batch
                smiles_batch = []
                smiles_batch_acq = []
                for candidate_smiles, acq in zip(acq_smiles, acq_vals):
                    if (
                        candidate_smiles not in gp_train_smiles_set
                        and candidate_smiles not in smiles_batch
                    ):
                        smiles_batch.append(candidate_smiles)
                        smiles_batch_acq.append(acq)
                    if len(smiles_batch) >= config["bo_batch_size"]:
                        break
                del candidate_smiles, acq

                assert (
                    len(smiles_batch) > 0
                ), "Empty batch, shouldn't happen. Must be problem with GA."

                smiles_batch_np, invalid_idx = smiles_to_fingerprint_stack(
                    smiles_batch, config, x_train_np.dtype
                )

                # Get predictions about SMILES batch before training on it
                smiles_batch_mu_pre, smiles_batch_var_pre = batch_predict_mu_var_numpy(
                    gp_model, torch.as_tensor(smiles_batch_np)
                )

                # Score these SMILES
                smiles_batch_scores = self.oracle(smiles_batch)

                # Add new points to GP training data
                gp_train_smiles_list += smiles_batch
                gp_train_smiles_set.update(gp_train_smiles_list)
                x_train_np = np.concatenate([x_train_np, smiles_batch_np], axis=0)
                y_invalid_offset = y_train_np.shape[0]
                y_train_np = np.concatenate(
                    [
                        y_train_np,
                        np.asarray(smiles_batch_scores, dtype=y_train_np.dtype),
                    ],
                    axis=0,
                )
                for i in invalid_idx:
                    y_train_np = np.delete(y_train_np, i + y_invalid_offset)
                    y_invalid_offset -= 1

                gp_model.set_train_data(
                    inputs=torch.as_tensor(x_train_np),
                    targets=torch.as_tensor(y_train_np),
                    strict=False,
                )

                # Get predictions about SMILES batch AFTER training on it
                (
                    smiles_batch_mu_post1,
                    smiles_batch_var_post1,
                ) = batch_predict_mu_var_numpy(
                    gp_model, torch.as_tensor(smiles_batch_np)
                )

                # Assemble full batch results
                batch_results = []
                i = 0
                for j, s in enumerate(smiles_batch):
                    if j not in invalid_idx:
                        transformed_score = self.oracle(s)
                        pred_dict = dict(
                            mu=float(smiles_batch_mu_pre[i]),
                            std=float(np.sqrt(smiles_batch_var_pre[i])),
                            acq=smiles_batch_acq[i],
                        )
                        pred_dict["pred_error_in_stds"] = (
                            pred_dict["mu"] - transformed_score
                        ) / pred_dict["std"]
                        pred_dict_post1 = dict(
                            mu=float(smiles_batch_mu_post1[i]),
                            std=float(np.sqrt(smiles_batch_var_post1[i])),
                        )
                        res = dict(
                            bo_iter=bo_iter,
                            smiles=s,
                            raw_score=self.oracle(s),
                            transformed_score=transformed_score,
                            predictions=pred_dict,
                            predictions_after_fit=pred_dict_post1,
                        )
                        batch_results.append(res)

                        del pred_dict, pred_dict_post1, res, transformed_score

                        i += 1
                    bo_query_res.extend(batch_results)
*/

 """