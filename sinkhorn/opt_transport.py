import numpy as np

from sinkhorn.regularization import entropy
from sinkhorn.sinkhorn import SinkhornParameters, sinkhorn_algorithm
from utils.param import ConvergenceMaxIterations


DEFAULT_NUM_ITERATIONS = 100


def entropy_regularized_opt_transport_dual_dist(
    r: np.ndarray,
    c: np.ndarray,
    M: np.ndarray,
    lmbda: float,
) -> np.ndarray:
    f"""
    Compute the entropy regularized optimal transport dual distance (d_M^lmbda in the paper).
    p.4 on https://papers.nips.cc/paper_files/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf
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
            num_iterations=DEFAULT_NUM_ITERATIONS
        )
    )

    K = np.exp(-M * lmbda)
    P = sinkhorn_algorithm(K, r, c, param)

    mat_loss = np.sum(P * M)
    print(f"M: {M}")
    print(f"P: {P}")
    print(f"P * M: {np.sum(P * M)}")
    print(f"mat_loss: {mat_loss}")
    print(f"log: {np.log(P)}")
    print(f"P: {P}")
    print(f"entropy(P): {entropy(P)}")

    return -entropy(P) / lmbda + np.sum(P * M)
