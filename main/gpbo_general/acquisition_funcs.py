import numpy as np
from scipy import stats
import functools


def upper_confidence_bound(mu: np.array, var: np.array, beta: float):
    return mu + np.sqrt(beta * var)


def expected_improvement(mu: np.array, var: np.array, y_best: float):
    std = np.sqrt(var)
    z = (mu - y_best) / std
    ei = (mu - y_best) * stats.norm.cdf(z) + std * stats.norm.pdf(z)

    # Bound EI at some small but non-zero value (corresponds to about 10 st devs)
    return np.maximum(ei, 1e-30)


def acq_f_of_time(bo_iter, bo_state_dict):
    # Beta log-uniform between ~0.3 and ~30
    # beta = 10 ** (x ~ Uniform(-0.5, 1.5))
    beta_curr = 10 ** float(np.random.uniform(-0.5, 1.5))
    return functools.partial(
        upper_confidence_bound,
        beta=beta_curr**2,  # due to different conventions of what beta is in UCB
    )
