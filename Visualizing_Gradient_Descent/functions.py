# -----------------------------
# ✅ functions.py
# -----------------------------
import numpy as np
from typing import Union

class FunctionDefinition:
    """
    Base class for mathematical functions used in gradient descent visualization.
    """
    name: str
    formula_latex: str
    has_minimum: bool = True
    recommended_lr: float = 0.1
    stationary_points: list[float] = []

    def evaluate(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def derivative(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError


class Quadratic(FunctionDefinition):
    name = "Quadratic"
    formula_latex = "$f(x) = x^2$"
    stationary_points = [0]
    recommended_lr = 0.1

    def evaluate(self, x):
        return x ** 2

    def derivative(self, x):
        return 2 * x


class Cubic(FunctionDefinition):
    name = "Cubic"
    formula_latex = "$f(x) = x^3$"
    has_minimum = False
    recommended_lr = 0.005
    stationary_points = [0]

    def evaluate(self, x):
        return x ** 3

    def derivative(self, x):
        return 3 * x ** 2


class Sine(FunctionDefinition):
    name = "Sine"
    formula_latex = "$f(x) = \\sin(x)$"
    stationary_points = [0, np.pi, 2*np.pi]
    recommended_lr = 0.05

    def evaluate(self, x):
        return np.sin(x)

    def derivative(self, x):
        return np.cos(x)


class ExponentialDecay(FunctionDefinition):
    name = "Exponential Decay"
    formula_latex = "$f(x) = e^{-x}$"
    stationary_points = []
    recommended_lr = 0.1

    def evaluate(self, x):
        return np.exp(-x)

    def derivative(self, x):
        return -np.exp(-x)


FUNCTIONS = {
    "Quadratic": Quadratic(),
    "Cubic": Cubic(),
    "Sine": Sine(),
    "Exponential Decay": ExponentialDecay()
}
