import os
import sys
import numpy as np
from rdkit.Chem import rdMolDescriptors
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))
from main.optimizer import BaseOptimizer
from utils import Variable, seq_to_smiles, unique
from model import RNN
from data_structs import Vocabulary, Experience
import torch
import random
import functools
import gpytorch
from main.gpbo.gp import (TanimotoGP, batch_predict_mu_var_numpy, 
    fit_gp_hyperparameters)
from main.gpbo.fingerprints import smiles_to_fp_array
from main.gpbo.bo import acquisition_funcs
from main.gpbo.function_utils import CachedFunction, CachedBatchFunction

def get_trained_gp(X_train, y_train,):

    # Fit model using type 2 maximum likelihood
    model = TanimotoGP(
        train_x=torch.as_tensor(X_train), train_y=torch.as_tensor(y_train)
    )
    fit_gp_hyperparameters(model)
    return model

def acq_f_of_time(bo_iter, bo_state_dict):
    # Beta log-uniform between ~0.3 and ~30
    # beta = 10 ** (x ~ Uniform(-0.5, 1.5))
    beta_curr = 10 ** float(np.random.uniform(-0.5, 1.5))
    return functools.partial(
        acquisition_funcs.upper_confidence_bound,
        beta=beta_curr ** 2,  # due to different conventions of what beta is in UCB
    )
    
class REINVENT_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "reinvent"

    def optimize_rnn(self, gp_model, acq_func_np, smiles_to_np_fingerprint, config):
        
        def _acq_func_smiles(smiles_list):
            #deal with invalid strings
            fp_array = list(map(smiles_to_np_fingerprint, smiles_list))
            fp_array_filtered = []

            reference = 0
            while fp_array[reference] is None:
                reference+=1
                if reference >= len(fp_array):
                    return 0
            for i in fp_array:
                if i is not None:
                    fp_array_filtered.append(i)
                else:
                    fp_array_filtered.append(np.ones_like(fp_array[reference]))
           # for f in fp_array_filtered:
           #     print(f.shape)
            fp_array = np.stack(fp_array_filtered)
            if gp_model.train_inputs[0].dtype == torch.float32:
                fp_array = fp_array.astype(np.float32)
            elif gp_model.train_inputs[0].dtype == torch.float64:
                fp_array = fp_array.astype(np.float64)
            else:
                raise ValueError(gp_model.train_inputs[0].dtype)
            mu_pred, var_pred = batch_predict_mu_var_numpy(
                gp_model, torch.as_tensor(fp_array), batch_size=2 ** 15
            )
            acq_vals = acq_func_np(mu_pred, var_pred)
            return list(map(float, acq_vals))

        scoring_function = CachedBatchFunction(_acq_func_smiles)

        path_here = os.path.dirname(os.path.realpath(__file__))
        restore_prior_from=os.path.join(path_here, 'data/Prior.ckpt')
        restore_agent_from=restore_prior_from 
        voc = Vocabulary(init_from_file=os.path.join(path_here, "data/Voc"))

        Prior = RNN(voc)
        Agent = RNN(voc)

        # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
        # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
        # to the CPU.
        if torch.cuda.is_available():
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here,'data/Prior.ckpt')))
            Agent.rnn.load_state_dict(torch.load(restore_agent_from))
        else:
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here, 'data/Prior.ckpt'), map_location=lambda storage, loc: storage))
            Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

        # We dont need gradients with respect to Prior
        for param in Prior.rnn.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=config['learning_rate'])

        # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
        # occur more often (which means the agent can get biased towards them). Using experience replay is
        # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
        experience = Experience(voc)

        print("Model initialized, starting training...")

        step = 0
        patience = 0

        while True:
            
            # Sample from Agent
            seqs, agent_likelihood, entropy = Agent.sample(config['batch_size'])

            # Remove duplicates, ie only consider unique seqs
            unique_idxs = unique(seqs)
            seqs = seqs[unique_idxs]
            agent_likelihood = agent_likelihood[unique_idxs]
            entropy = entropy[unique_idxs]

            # Get prior likelihood and score
            prior_likelihood, _ = Prior.likelihood(Variable(seqs))
            smiles = seq_to_smiles(seqs, voc)
            score = np.array(scoring_function(smiles, batch = True))

            if self.finish:
                print('max oracle hit')
                break 

            # Calculate augmented likelihood
            augmented_likelihood = prior_likelihood.float() + config['sigma'] * Variable(score).float()
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

            # Experience Replay
            # First sample
            if config['experience_replay'] and len(experience)>config['experience_replay']:
                exp_seqs, exp_score, exp_prior_likelihood = experience.sample(config['experience_replay'])
                exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
                exp_augmented_likelihood = exp_prior_likelihood + config['sigma'] * exp_score
                exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
                loss = torch.cat((loss, exp_loss), 0)
                agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

            # Then add new experience
            prior_likelihood = prior_likelihood.data.cpu().numpy()
            new_experience = zip(smiles, score, prior_likelihood)
            experience.add_experience(new_experience)

            # Calculate loss
            loss = loss.mean()

            # Add regularizer that penalizes high likelihood for the entire sequence
            loss_p = - (1 / agent_likelihood).mean()
            loss += 5 * 1e3 * loss_p

            # Calculate gradients and make an update to the network weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Convert to numpy arrays so that we can print them
            augmented_likelihood = augmented_likelihood.data.cpu().numpy()
            agent_likelihood = agent_likelihood.data.cpu().numpy()

            step += 1

        smiles_2_acq_dict = scoring_function.cache

        # Sort and return results (highest acq func first)
        sm_ac_list = list(smiles_2_acq_dict.items())
        sm_ac_list.sort(reverse=True, key=lambda t: t[1])
        smiles_out = [s for s, v in sm_ac_list]
        acq_out = [v for s, v in sm_ac_list]
        return smiles_out, acq_out

    def _optimize(self, oracle, config):
        # Functions to do retraining
        def get_inducing_indices(y):
            """
            To reduce the training cost of GP model, we only select 
            top-n_train_gp_best and n_train_gp_rand random samples from data.
            """
            argsort = np.argsort(-y)  # Biggest first
            best_idxs = list(argsort[: config['n_train_gp_best']])
            remaining_idxs = list(argsort[config['n_train_gp_best'] :])
            if len(remaining_idxs) <= config['n_train_gp_rand']:
                rand_idxs = remaining_idxs
            else:
                rand_idxs = random.sample(remaining_idxs, k=config['n_train_gp_rand'])
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
            starting_population = self.all_smiles[:config["initial_population_size"]]
        else:
            # Exploration run
            starting_population = np.random.choice(self.all_smiles, config["initial_population_size"])

        smiles_pool = set(starting_population)

        # Create data; fit exact GP
        fingerprint_func = functools.partial(
            rdMolDescriptors.GetMorganFingerprintAsBitVect,
            radius=int(config['fp_radius']),
            nBits=int(config['fp_nbits'])
        )
        smiles_to_np_fingerprint = functools.partial(
            smiles_to_fp_array, fingerprint_func=fingerprint_func
        )
        gp_train_smiles = list(smiles_pool)
        x_train = np.stack([smiles_to_np_fingerprint(s) for s in gp_train_smiles]).astype(NP_DTYPE)
        values = self.oracle(gp_train_smiles)
        y_train = np.asarray(values).astype(NP_DTYPE)

        ind_idx_start = get_inducing_indices(y_train)
        x_train = torch.as_tensor(x_train)
        y_train = torch.as_tensor(y_train)
        gp_model = get_trained_gp(
            x_train[ind_idx_start], y_train[ind_idx_start]
        )

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
            num_train_data_not_known = len(gp_train_smiles_set - set(start_cache.keys()))
            gp_train_smiles_list = list(gp_train_smiles_set)
            gp_train_smiles_scores = self.oracle(list(gp_train_smiles_set))

            # Store GP training data
            x_train_np = np.stack(
                list(map(smiles_to_np_fingerprint, gp_train_smiles_list))
            ).astype(NP_DTYPE)
            y_train_np = np.array(gp_train_smiles_scores).astype(NP_DTYPE)
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
            refit_gp_change_subset(bo_iter=0, gp_model=gp_model, bo_state_dict=bo_state_dict)

            # Actual BO loop
            for bo_iter in range(1, config['max_bo_iter'] + 1):

                if self.finish: 
                    break 

                # Current acquisition function
                curr_acq_func = acq_f_of_time(bo_iter, bo_state_dict)

                # Optimize acquisition function
                print("Maximizing BO surrogate...")
                acq_smiles, acq_vals = self.optimize_rnn(
                    gp_model=gp_model,
                    acq_func_np=curr_acq_func,
                    smiles_to_np_fingerprint=smiles_to_np_fingerprint,
                    config=config
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
                    if len(smiles_batch) >= config['bo_batch_size']:
                        break
                del candidate_smiles, acq

                assert (
                    len(smiles_batch) > 0
                ), "Empty batch, shouldn't happen. Must be problem with GA."
                
                smiles_batch_np = np.stack(
                    list(map(smiles_to_np_fingerprint, smiles_batch))
                ).astype(x_train_np.dtype)

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
                y_train_np = np.concatenate(
                    [y_train_np, np.asarray(smiles_batch_scores, dtype=y_train_np.dtype)],
                    axis=0,
                )
                gp_model.set_train_data(
                    inputs=torch.as_tensor(x_train_np),
                    targets=torch.as_tensor(y_train_np),
                    strict=False,
                )

                # Get predictions about SMILES batch AFTER training on it
                smiles_batch_mu_post1, smiles_batch_var_post1 = batch_predict_mu_var_numpy(
                    gp_model, torch.as_tensor(smiles_batch_np)
                )

                # Assemble full batch results
                batch_results = []
                for i, s in enumerate(smiles_batch):
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
                bo_query_res.extend(batch_results)
