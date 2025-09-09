import numpy as np
from utils.param import ConvergenceMaxIterations, ConvergenceTolerance
from dataclasses import dataclass

import logging

logger = logging.getLogger(__name__)


@dataclass
class SinkhornParameters:
    convergence_criteria: ConvergenceMaxIterations | ConvergenceTolerance
    num_iterations_to_log: int = 100


def assert_doubly_stochastic(mat: np.ndarray):
    assert len(mat.shape) == 2, "Matrix must be 2D"
    assert mat.shape[0] == mat.shape[1], "Matrix must be square"

    assert np.allclose(mat.sum(axis=0), 1), "Row sums must be 1"
    assert np.allclose(mat.sum(axis=1), 1), "Column sums must be 1"

    pass


def sinkhorn_rescale(mat: np.ndarray) -> np.ndarray:
    """
    Perform one Sinkhorn iteration. Simply scale the rows and columns (in that sequence) of the matrix to sum up to 1.
    param:
        mat: np.ndarray, the input matrix
    return:
        np.ndarray, the scaled matrix after one Sinkhorn iteration
    """
    row_sums = mat.sum(axis=1)
    mat /= row_sums[np.newaxis, :]
    col_sums = mat.sum(axis=0)
    mat /= col_sums[:, np.newaxis]
    return mat


def sinkhorn_algorithm(
    mat: np.ndarray, r: np.ndarray, c: np.ndarray, parameters: SinkhornParameters
) -> np.ndarray:
    """
    Perform the Sinkhorn algorithm.
    Original paper: https://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-s.pdf.
    Cuturi's paper on application to optimal transport: https://papers.nips.cc/paper_files/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf
    param:
        mat: np.ndarray, the input matrix
        parameters: SinkhornParameters, the parameters for the Sinkhorn algorithm
    return:
        np.ndarray
    """
    assert len(mat.shape) == 2, "Matrix must be 2D"
    assert mat.shape[0] == len(
        r
    ), "Matrix must have the same number of rows as the number of elements in r"
    assert mat.shape[1] == len(
        c
    ), "Matrix must have the same number of columns as the number of elements in c"
    assert mat.shape[0] == mat.shape[1], "Matrix must be square"

    num_iterations = 0

    # TODO better initialization
    u, v = np.random.rand(mat.shape[0]), np.random.rand(mat.shape[1])
    old_u, old_v = None, None

    while True:
        (u, v, old_u, old_v) = (r / mat @ v, c / mat.T @ u, u, v)
        num_iterations += 1

        if num_iterations % parameters.num_iterations_to_log == 0:
            logger.debug(f"Sinkhorn iteration {num_iterations}: {mat}")

        match parameters.convergence_criteria:
            case ConvergenceMaxIterations(num_iterations=max_num_iterations):
                if num_iterations >= max_num_iterations:
                    break
            case ConvergenceTolerance(err=err):
                # TODO: implement better convergence criteria in Cuturi's paper
                if np.linalg.norm(u - old_u) < err and np.linalg.norm(v - old_v) < err:
                    break

    return mat
