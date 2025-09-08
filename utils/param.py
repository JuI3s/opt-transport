from dataclasses import dataclass

@dataclass
class ConvergenceMaxIterations:
    num_iterations: int

@dataclass
class ConvergenceTolerance:
    err: float
    