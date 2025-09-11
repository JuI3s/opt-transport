"""
Test the Sinkhorn algorithm.
"""

from copy import deepcopy

import numpy as np
import torch
from sinkhorn.sinkhorn import (
    SinkhornParameters,
    sinkhorn_algorithm,
    sinkhorn_algorithm_torch,
    sinkhorn_rescale,
    assert_doubly_stochastic,
)
from utils.param import ConvergenceMaxIterations


def test_sinkhorn_is_fixed_point_for_doubly_stochastic_matrix():
    """
    Test that the Sinkhorn algorithm is a fixed point for a doubly stochastic matrix.
    """
    a = np.array([[1.0, 0], [0, 1.0]])
    assert_doubly_stochastic(a)
    b = sinkhorn_rescale(deepcopy(a))
    assert_doubly_stochastic(b)
    assert np.allclose(a, b)


def test_sinkhorn_algorithm_torch():
    """
    Test the Sinkhorn algorithm.
    """
    # Generate a random cost matrix M and random positive vectors r and c that sum to 1
    n = 4
    torch.manual_seed(0)
    np.random.seed(0)
    cost_mat = torch.rand((n, n))
    r = torch.rand(n)
    c = torch.rand(n)
    param = SinkhornParameters(
        convergence_criteria=ConvergenceMaxIterations(num_iterations=100)
    )

    res_torch = sinkhorn_algorithm_torch(cost_mat, r, c, param)
    res_np = sinkhorn_algorithm(cost_mat.numpy(), r.numpy(), c.numpy(), param)

    assert res_torch.shape == res_np.shape
    assert res_torch.shape == cost_mat.shape
    assert np.allclose(res_torch.numpy(), res_np, atol=1e-4)
