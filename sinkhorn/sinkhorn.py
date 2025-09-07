import numpy as np


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


def sinkhorn_algorithm(mat: np.ndarray, max_iterations: int = 1000) -> np.ndarray:
    for _ in range(max_iterations):
        mat = sinkhorn_iteration(mat)
    return mat
