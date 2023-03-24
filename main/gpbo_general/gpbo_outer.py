import os
import sys
import numpy as np
from rdkit.Chem import rdMolDescriptors
import heapq

path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append("/".join(path_here.rstrip("/").split("/")[:-2]))
from main.optimizer import BaseOptimizer
import torch
import random
import functools
import gpytorch
from gp import (
    TanimotoGP,
    batch_predict_mu_var_numpy,
    fit_gp_hyperparameters,
)
from gp.gp_utils import get_trained_gp
from fingerprints import smiles_to_fingerprint_stack, smiles_to_fp_array
from acquisition_funcs import acq_f_of_time
from function_utils import CachedFunction, CachedBatchFunction
from main.utils.choose_optimizer import choose_optimizer


class GPBO_Optimizer(BaseOptimizer):
    def __init__(self, optimizer, args=None):
        super().__init__(args)
        self.model_name = "gpbo_" + optimizer
        self.internal_optimizer = choose_optimizer(method=optimizer, args=args)

    def _acq_func_smiles(self, smiles_list, gp_model, acq_func_np, config):
        fp_array, invalid_idx = smiles_to_fingerprint_stack(smiles_list, config)

        if gp_model.train_inputs[0].dtype == torch.float32:
            fp_array = fp_array.astype(np.float32)
        elif gp_model.train_inputs[0].dtype == torch.float64:
            fp_array = fp_array.astype(np.float64)
        else:
            raise ValueError(gp_model.train_inputs[0].dtype)
        mu_pred, var_pred = batch_predict_mu_var_numpy(
            gp_model, torch.as_tensor(fp_array), batch_size=2**15
        )
        acq_vals = acq_func_np(mu_pred, var_pred)
        acq_list = list(map(float, acq_vals))
        for i in invalid_idx:
            acq_list = acq_list[:i] + [0] + acq_list[i:]
        return acq_list

    def _optimize(self, oracle, config):
        # Functions to do retraining
        def get_inducing_indices(y):
            """
            To reduce the training cost of GP model, we only select
            top-n_train_gp_best and n_train_gp_rand random samples from data.
            """
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
         #   if smiles_pool is None:
          #      smiles_pool = set()
          #  else:
        #        smiles_pool = set(smiles_pool)
            # smiles_pool.update(start_cache.keys())
            # smiles_pool.update(gp_train_smiles_set)
            # assert (
            #     len(smiles_pool) > 0
            # ), "No SMILES were provided to the algorithm as training data, known scores, or a SMILES pool."

            # Handle edge case of no training data
            # if len(gp_train_smiles_set) == 0:
            #     #     f"No SMILES were provided to train GP. A random one will be chosen from the pool to start training."
            #     random_smiles = random.choice(list(smiles_pool))
            #     gp_train_smiles_set.add(random_smiles)
            #     del random_smiles

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
         #   carryover_smiles_pool = set()
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
         #       smiles_pool.update(acq_smiles)

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
