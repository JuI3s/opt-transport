"""
General parameters.
"""

from dataclasses import dataclass


@dataclass
class ConvergenceMaxIterations:
    """
    Convergence criteria based on the number of iterations.
    """
    num_iterations: int


@dataclass
class ConvergenceTolerance:
    """
    Convergence criteria based on the tolerance.
    """
    err: float
