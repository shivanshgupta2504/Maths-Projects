import numpy as np
from typing import Callable, Protocol

class Function2D:
    """
    Base class for 2D functions used in gradient descent.
    """

    name: str
    formula_latex: str
    recommended_lr: float = 0.1
    has_minimum: bool = True

    def evaluate(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        raise NotImplementedError

    def gradient(self, x: float, y: float) -> tuple[float, float]:
        raise NotImplementedError


class QuadraticBowl(Function2D):
    name = "Quadratic Bowl"
    formula_latex = "$f(x, y) = x^2 + y^2$"
    recommended_lr = 0.1
    has_minimum = True

    def evaluate(self, x, y):
        return x ** 2 + y ** 2

    def gradient(self, x, y):
        return 2 * x, 2 * y


class Saddle(Function2D):
    name = "Saddle"
    formula_latex = "$f(x, y) = x^2 - y^2$"
    recommended_lr = 0.05
    has_minimum = False

    def evaluate(self, x, y):
        return x ** 2 - y ** 2

    def gradient(self, x, y):
        return 2 * x, -2 * y


class SinCos(Function2D):
    name = "Sin + Cos"
    formula_latex = "$f(x, y) = \\sin(x) + \\cos(y)$"
    recommended_lr = 0.05
    has_minimum = False

    def evaluate(self, x, y):
        return np.sin(x) + np.cos(x)

    def gradient(self, x, y):
        return np.cos(x), -np.sin(y)

# Registry of Available 2D Functions
FUNCTIONS_2D = {
    "Quadratic Bowl": QuadraticBowl(),
    "Saddle": Saddle(),
    "Sin + Cos": SinCos(),
}
