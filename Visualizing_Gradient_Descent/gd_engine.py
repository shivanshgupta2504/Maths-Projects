# -----------------------------
# ✅ gd_engine.py
# -----------------------------
import numpy as np
from typing import Callable

class Function1D:
    def __init__(self, func: Callable, derivative: Callable) -> None:
        self.func = func
        self.derivative_func = derivative

    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        return self.func(x)

    def gradient(self, x: float | np.ndarray) -> float | np.ndarray:
        return self.derivative_func(x)


class GradientDescent:
    def __init__(self, func: Function1D, start_x: float, lr: float, steps: int) -> None:
        self.func = func
        self.start_x = start_x
        self.lr = lr
        self.steps = steps
        self.history = []

    def run(self) -> list[float]:
        try:
            x = self.start_x
            self.history = [x]
            for _ in range(self.steps):
                grad = self.func.gradient(x)
                grad = np.clip(grad, -1e5, 1e5)  # ⛑️ prevent overflow
                x -= self.lr * grad
                x = np.clip(x, -1e8, 1e8)        # ⛑️ prevent runaway
                self.history.append(x)
            return self.history
        except Exception as e:
            print(f"Error during gradient descent: {e}")
            return []