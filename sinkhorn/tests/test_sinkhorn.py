"""
Test the Sinkhorn algorithm.
"""

from copy import deepcopy

import numpy as np
from sinkhorn.sinkhorn import sinkhorn_rescale, assert_doubly_stochastic


def test_sinkhorn_is_fixed_point_for_doubly_stochastic_matrix():
    """
    Test that the Sinkhorn algorithm is a fixed point for a doubly stochastic matrix.
    """
    a = np.array([[1.0, 0], [0, 1.0]])
    assert_doubly_stochastic(a)
    b = sinkhorn_rescale(deepcopy(a))
    assert_doubly_stochastic(b)
    assert np.allclose(a, b)
