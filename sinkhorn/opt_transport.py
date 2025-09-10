"""
Optimal transport based distance.
"""

import numpy as np

from sinkhorn.sinkhorn import SinkhornParameters, sinkhorn_algorithm
from utils.param import ConvergenceMaxIterations


_DEFAULT_NUM_ITERATIONS = 20


def entropy_regularized_opt_transport_dual_dist(
    r: np.ndarray,
    c: np.ndarray,
    m: np.ndarray,
    lmbda: float,
) -> np.ndarray:
    """
    Compute the entropy regularized optimal transport dual distance (d_M^lmbda in the paper).
    p.4 on
    https://papers.nips.cc/paper_files/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf
    P^lmbda = argmin_P <P, M> - 1/lmbda * h(P)
    param:
        r: np.ndarray, the source distribution
        c: np.ndarray, the target distribution
        M: np.ndarray, the cost matrix
        lmbda: float, the regularization parameter
    return:
        np.ndarray, the entropy regularized optimal transport dual distance
    """
    param = SinkhornParameters(
        convergence_criteria=ConvergenceMaxIterations(
            num_iterations=_DEFAULT_NUM_ITERATIONS
        )
    )

    r_non_zero_indices = np.argwhere(r > 0).squeeze()
    c_non_zero_indices = np.argwhere(c > 0).squeeze()

    select_m = m[np.ix_(r_non_zero_indices, c_non_zero_indices)]
    k_mat = np.exp(-select_m * lmbda)
    p_mat = sinkhorn_algorithm(
        k_mat, r[r_non_zero_indices], c[c_non_zero_indices], param
    )
    assert (
        p_mat.shape == k_mat.shape
    ), f"p_mat shape: {p_mat.shape}, k_mat shape: {k_mat.shape}"

    return np.sum(p_mat * select_m)
