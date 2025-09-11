"""
Sinkhorn algorithm.
"""

from dataclasses import dataclass
import logging


import numpy as np
from utils.param import ConvergenceMaxIterations, ConvergenceTolerance


logger = logging.getLogger(__name__)


@dataclass
class SinkhornParameters:
    """
    Parameters for the Sinkhorn algorithm.
    """

    convergence_criteria: ConvergenceMaxIterations | ConvergenceTolerance
    num_iterations_to_log: int = 50


def assert_doubly_stochastic(mat: np.ndarray):
    """
    Assert that the matrix is doubly stochastic.
    """
    assert len(mat.shape) == 2, "Matrix must be 2D"
    assert mat.shape[0] == mat.shape[1], "Matrix must be square"

    assert np.allclose(mat.sum(axis=0), 1), "Row sums must be 1"
    assert np.allclose(mat.sum(axis=1), 1), "Column sums must be 1"


def sinkhorn_rescale(mat: np.ndarray) -> np.ndarray:
    """
    Perform one Sinkhorn iteration. Simply scale the rows and columns
    (in that sequence) of the matrix to sum up to 1.
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
    Cuturi's paper on application to optimal transport:
    https://papers.nips.cc/paper_files/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf
    param:
        mat: np.ndarray, the input matrix
        r: np.ndarray, the source distribution which has positive elements
        c: np.ndarray, the target distribution which has positive elements
        parameters: SinkhornParameters, the parameters for the Sinkhorn algorithm
    return:
        np.ndarray
    """
    assert np.all(r > 0), "r must have positive elements"
    assert np.all(c > 0), "c must have positive elements"
    assert len(mat.shape) == 2, "Matrix must be 2D"
    assert mat.shape[0] == len(
        r
    ), f"""Matrix must have the same number of rows as the number of elements in r,
    {mat.shape[0]} != {len(r)}"""
    assert mat.shape[1] == len(
        c
    ), f"""Matrix must have the same number of columns as the number of elements in c,
    {mat.shape[1]} != {len(c)}"""

    num_iterations = 0
    u, old_u = np.ones(mat.shape[0]), None

    while True:
        mat_tilde = np.diag(1.0 / r) @ mat
        u, old_u = 1 / (mat_tilde @ (c / (mat.T @ u))), u

        num_iterations += 1

        if num_iterations % parameters.num_iterations_to_log == 0:
            logger.debug("Sinkhorn iteration: %s, u: %s", num_iterations, u)

        match parameters.convergence_criteria:
            case ConvergenceMaxIterations(num_iterations=max_num_iterations):
                if num_iterations >= max_num_iterations:
                    break
            case ConvergenceTolerance(err=err):
                # TODO: implement better convergence criteria in Cuturi's paper
                if np.linalg.norm(u - old_u) < err:
                    break

    v = c / (mat.T @ u)
    return np.diag(u) @ mat @ np.diag(v)
