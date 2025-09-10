"""
Regularization terms for optimal transport.
"""

import numpy as np

# Implement the regularization terms in Cuturi's paper on application to optimal transport:
# https://papers.nips.cc/paper_files/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf


def entropy(x: np.ndarray):
    """
    Entropy regularization term.
    """
    return -np.sum(x * np.log(x))


def kl_div(x: np.ndarray, y: np.ndarray):
    """
    Kullback-Leibler divergence regularization term.
    """
    return np.sum(x * np.log(x / y))


def r_and_c_from_transport_matrix(mat: np.ndarray):
    """
    Get the source and target distributions from the transport matrix.
    """
    return mat.sum(axis=0), mat.sum(axis=1)
