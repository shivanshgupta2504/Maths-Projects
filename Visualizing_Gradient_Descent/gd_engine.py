# -----------------------------
# ✅ gd_engine.py (2D Extension)
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


class GradientDescent2D:
    def __init__(self, func, start: tuple[float, float], lr: float, steps: int) -> None:
        self.func = func
        self.start_x, self.start_y = start
        self.lr = lr
        self.steps = steps
        self.path = []

    def run(self) -> list[tuple[float, float]]:
        try:
            x, y = self.start_x, self.start_y
            self.path = [(x, y)]

            for _ in range(self.steps):
                dx, dy = self.func.gradient(x, y)
                dx = np.clip(dx, -1e2, 1e2)
                dy = np.clip(dy, -1e2, 1e2)
                x -= self.lr * dx
                y -= self.lr * dy
                x = np.clip(x, -1e8, 1e8)
                y = np.clip(y, -1e8, 1e8)
                self.path.append((x, y))

            return self.path
        except Exception as e:
            print(f"Error in 2D gradient descent: {e}")
            return []
