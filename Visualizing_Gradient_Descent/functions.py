import numpy as np
from typing import Callable

class FunctionDefinition:
    """
    Base class for mathematical functions used in gradient descent visualization.
    """
    name: str
    formula_latex: str

    def evaluate(self, x: float) -> float:
        raise NotImplementedError

    def derivative(self, x: float) -> float:
        raise NotImplementedError


class Quadratic(FunctionDefinition):
    name = "Quadratic"
    formula_latex = "$f(x) = x^2$"

    def evaluate(self, x: float) -> float:
        return x ** 2

    def derivative(self, x: float) -> float:
        return 2 * x


class Cubic(FunctionDefinition):
    name = "Cubic"
    formula_latex = "$f(x) = x^3$"

    def evaluate(self, x: float) -> float:
        return x ** 3

    def derivative(self, x: float) -> float:
        return 3 * x ** 2


class Sine(FunctionDefinition):
    name = "Sine"
    formula_latex = "$f(x) = \sin(x)$"

    def evaluate(self, x: float) -> float:
        return np.sin(x)

    def derivative(self, x: float) -> float:
        return np.cos(x)


class ExponentialDecay(FunctionDefinition):
    name = "Exponential Decay"
    formula_latex = "$f(x) = e^{-x}$"

    def evaluate(self, x: float) -> float:
        return np.exp(-x)

    def derivative(self, x: float) -> float:
        return -np.exp(-x)


# For dropdown access
FUNCTIONS = {
    "Quadratic": Quadratic(),
    "Cubic": Cubic(),
    "Sine": Sine(),
    "Exponential Decay": ExponentialDecay()
}
