import numpy as np
from typing import Callable, List

class Function1D:
    """
    Represents a 1D function and its derivative.
    """

    def __init__(self, func: Callable[[float], float], derivative: Callable[[float], float]) -> None:
        self.func = func
        self.derivative = derivative

    def evaluate(self, x: float) -> float:
        """
        Evaluate the function at x.
        """
        return self.func(x)

    def gradient(self, x: float) -> float:
        """
        Compute the derivative at x.
        """
        return self.derivative(x)


class GradientDescent:
    """
    Performs gradient descent on a given 1D function.
    """

    def __init__(self, func: Function1D, start_x: float, lr: float, steps: int) -> None:
        self.func = func
        self.start_x = start_x
        self.lr = lr
        self.steps = steps
        self.history: List[float] = []

    def run(self) -> List[float]:
        """
        Run the gradient descent algorithm.

        Returns:
            List of x values visited during descent.
        """
        try:
            x = self.start_x
            self.history = [x]

            for _ in range(self.steps):
                grad = self.func.gradient(x)
                x -= self.lr * grad
                self.history.append(x)

            return self.history
        except Exception as e:
            print(f"Error during gradient descent: {e}")
            return []
