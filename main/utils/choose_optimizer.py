def choose_optimizer(method=None, args=None):
    if not method:
        method = args.method
    if method == "screening":
        from main.screening.run import Exhaustive_Optimizer as Optimizer
    elif method == "molpal":
        from main.molpal.run import MolPAL_Optimizer as Optimizer
    elif method == "graph_ga":
        from main.graph_ga.run import GB_GA_Optimizer as Optimizer
    elif method == "smiles_ga":
        from main.smiles_ga.run import SMILES_GA_Optimizer as Optimizer
    elif method == "selfies_ga":
        from main.selfies_ga.run import SELFIES_GA_Optimizer as Optimizer
    elif method == "synnet":
        from main.synnet.run import SynNet_Optimizer as Optimizer
    elif method == "hebo":
        from main.hebo.run import HEBO_Optimizer as Optimizer
    elif method == "graph_mcts":
        from main.graph_mcts.run import Graph_MCTS_Optimizer as Optimizer
    elif method == "smiles_ahc":
        from main.smiles_ahc.run import AHC_Optimizer as Optimizer
    elif method == "smiles_lstm_hc":
        from main.smiles_lstm_hc.run import SMILES_LSTM_HC_Optimizer as Optimizer
    elif method == "selfies_lstm_hc":
        from main.selfies_lstm_hc.run import SELFIES_LSTM_HC_Optimizer as Optimizer
    elif method == "dog_gen":
        from main.dog_gen.run import DoG_Gen_Optimizer as Optimizer
    elif method == "gegl":
        from main.gegl.run import GEGL_Optimizer as Optimizer
    elif method == "boss":
        from main.boss.run import BOSS_Optimizer as Optimizer
    elif method == "chembo":
        from main.chembo.run import ChemBOoptimizer as Optimizer
    elif method == "gpbo":
        from main.gpbo.run import GPBO_Optimizer as Optimizer
    elif method == "stoned":
        from main.stoned.run import Stoned_Optimizer as Optimizer
    elif method == "selfies_vae":
        from main.selfies_vae.run import SELFIES_VAEBO_Optimizer as Optimizer
    elif method == "smiles_vae":
        from main.smiles_vae.run import SMILES_VAEBO_Optimizer as Optimizer
    elif method == "jt_vae":
        from main.jt_vae.run import JTVAE_BO_Optimizer as Optimizer
    elif method == "dog_ae":
        from main.dog_ae.run import DoG_AE_Optimizer as Optimizer
    elif method == "pasithea":
        from main.pasithea.run import Pasithea_Optimizer as Optimizer
    elif method == "dst":
        from main.dst.run import DST_Optimizer as Optimizer
    elif method == "molgan":
        from main.molgan.run import MolGAN_Optimizer as Optimizer
    elif method == "mars":
        from main.mars.run import MARS_Optimizer as Optimizer
    elif method == "mimosa":
        from main.mimosa.run import MIMOSA_Optimizer as Optimizer
    elif method == "gflownet":
        from main.gflownet.run import GFlowNet_Optimizer as Optimizer
    elif method == "gflownet_al":
        from main.gflownet_al.run import GFlowNet_AL_Optimizer as Optimizer
    elif method == "moldqn":
        from main.moldqn.run import MolDQN_Optimizer as Optimizer
    elif method == "reinvent":
        from main.reinvent.run import REINVENT_Optimizer as Optimizer
    elif method == "reinvent_selfies":
        from main.reinvent_selfies.run import REINVENT_SELFIES_Optimizer as Optimizer
    elif method == "graphinvent":
        from main.graphinvent.run import GraphInvent_Optimizer as Optimizer
    elif method == "rationale_rl":
        from main.rationale_rl.run import Rationale_RL_Optimizer as Optimizer
    elif method == "gpbo_reinvent":
        from main.gpbo_reinvent.run import REINVENT_Optimizer as Optimizer
    elif "gpbo_" in method:
        from main.gpbo_general.gpbo_outer import GPBO_Optimizer as Optimizer

        internal_optimizer = "_".join(method.split("_")[1:])
        return Optimizer(optimizer=internal_optimizer, args=args)

    else:
        raise ValueError("Unrecognized method name.")

    return Optimizer(args=args)
