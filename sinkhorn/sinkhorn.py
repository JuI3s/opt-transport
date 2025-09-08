import numpy as np
from utils.param import ConvergenceMaxIterations, ConvergenceTolerance

class SinkhornParameters:
    convergence_criteria: ConvergenceMaxIterations | ConvergenceTolerance
    


def assert_doubly_stochastic(mat: np.ndarray):
    assert len(mat.shape) == 2, "Matrix must be 2D"
    assert mat.shape[0] == mat.shape[1], "Matrix must be square"

    assert np.allclose(mat.sum(axis=0), 1), "Row sums must be 1"
    assert np.allclose(mat.sum(axis=1), 1), "Column sums must be 1"

    pass


def sinkhorn_iteration(mat: np.ndarray) -> np.ndarray:
    row_sums = mat.sum(axis=1)
    mat /= row_sums[np.newaxis, :]
    col_sums = mat.sum(axis=0)
    mat /= col_sums[:, np.newaxis]
    return mat


def sinkhorn_algorithm(mat: np.ndarray, parameters: SinkhornParameters) -> np.ndarray:
    num_iterations = 0

    while True:
        mat_old, mat = mat, sinkhorn_iteration(mat)
        num_iterations += 1

        match parameters.convergence_criteria:
            case ConvergenceMaxIterations(num_iterations=max_num_iterations):
                if num_iterations >= max_num_iterations:
                    break
            case ConvergenceTolerance(err=err):
                if np.linalg.norm(mat - mat_old) < err:
                    break

    return mat
