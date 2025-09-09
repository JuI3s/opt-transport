import pytest   
import numpy as np
from sinkhorn.sinkhorn import sinkhorn_rescale, assert_doubly_stochastic
from copy import deepcopy

def test_sinkhorn_is_fixed_point_for_doubly_stochastic_matrix():
    A = np.array([[1.0, 0], [0, 1.0]])
    assert_doubly_stochastic(A)
    B = sinkhorn_rescale(deepcopy(A))
    assert_doubly_stochastic(B)
    assert np.allclose(A, B)
