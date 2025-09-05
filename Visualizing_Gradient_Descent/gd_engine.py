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
    def __init__(
        self,
        func: Function1D,
        start_x: float,
        lr: float,
        steps: int,
        optimizer_type: str = "Gradient Descent",
        optimizer_params: dict = None
    ) -> None:
        self.func = func
        self.start_x = start_x
        self.lr = lr
        self.steps = steps
        self.history = []
        self.optimizer = optimizer_type
        self.optimizer_params = optimizer_params or {}

        # Params with default fallback
        self.momentum = self.optimizer_params.get("momentum", 0.9)
        self.beta1 = self.optimizer_params.get("beta1", 0.9)
        self.beta2 = self.optimizer_params.get("beta2", 0.999)
        self.epsilon = self.optimizer_params.get("epsilon", 1e-8)
        self.decay_rate = self.optimizer_params.get("decay_rate", 0.9)

        self.v = 0
        self.s = 0
        self.m = 0

    def run(self) -> list[float]:
        try:
            x = self.start_x
            self.history = [x]

            for t in range(1, self.steps + 1):
                grad = self.func.gradient(x)
                grad = np.clip(grad, -1e5, 1e5)

                if self.optimizer == "Gradient Descent":
                    x -= self.lr * grad

                elif self.optimizer == "Momentum":
                    self.v = self.momentum * self.v - self.lr * grad
                    x += self.v

                elif self.optimizer == "RMSprop":
                    self.s = self.decay_rate * self.s + (1 - self.decay_rate) * grad ** 2
                    x -= (self.lr / (np.sqrt(self.s) + self.epsilon)) * grad

                elif self.optimizer == "Adam":
                    self.m = self.beta1 * self.m + (1 - self.beta1) * grad
                    self.s = self.beta2 * self.s + (1 - self.beta2) * grad ** 2
                    m_hat = self.m / (1 - self.beta1 ** t)
                    s_hat = self.s / (1 - self.beta2 ** t)
                    x -= (self.lr * m_hat) / (np.sqrt(s_hat) + self.epsilon)

                x = np.clip(x, -1e8, 1e8)
                self.history.append(x)

            return self.history
        except Exception as e:
            print(f"Error during gradient descent: {e}")
            return []


class GradientDescent2D:
    def __init__(
        self,
        func,
        start: tuple[float, float],
        lr: float,
        steps: int,
        optimizer_type: str = "Gradient Descent",
        optimizer_params: dict = None
    ) -> None:
        self.func = func
        self.start_x, self.start_y = start
        self.lr = lr
        self.steps = steps
        self.path = []
        self.optimizer = optimizer_type
        self.optimizer_params = optimizer_params or {}

        # Optimizer parameters
        self.momentum = self.optimizer_params.get("momentum", 0.9)
        self.beta1 = self.optimizer_params.get("beta1", 0.9)
        self.beta2 = self.optimizer_params.get("beta2", 0.999)
        self.epsilon = self.optimizer_params.get("epsilon", 1e-8)
        self.decay_rate = self.optimizer_params.get("decay_rate", 0.9)

        self.vx = self.vy = 0
        self.mx = self.my = 0
        self.sx = self.sy = 0

    def run(self) -> list[tuple[float, float]]:
        try:
            x, y = self.start_x, self.start_y
            self.path = [(x, y)]

            for t in range(1, self.steps + 1):
                dx, dy = self.func.gradient(x, y)
                dx = np.clip(dx, -1e2, 1e2)
                dy = np.clip(dy, -1e2, 1e2)

                if self.optimizer == "Gradient Descent":
                    x -= self.lr * dx
                    y -= self.lr * dy

                elif self.optimizer == "Momentum":
                    self.vx = self.momentum * self.vx - self.lr * dx
                    self.vy = self.momentum * self.vy - self.lr * dy
                    x += self.vx
                    y += self.vy

                elif self.optimizer == "RMSprop":
                    self.sx = self.decay_rate * self.sx + (1 - self.decay_rate) * dx ** 2
                    self.sy = self.decay_rate * self.sy + (1 - self.decay_rate) * dy ** 2
                    x -= (self.lr / (np.sqrt(self.sx) + self.epsilon)) * dx
                    y -= (self.lr / (np.sqrt(self.sy) + self.epsilon)) * dy

                elif self.optimizer == "Adam":
                    self.mx = self.beta1 * self.mx + (1 - self.beta1) * dx
                    self.my = self.beta1 * self.my + (1 - self.beta1) * dy
                    self.sx = self.beta2 * self.sx + (1 - self.beta2) * dx ** 2
                    self.sy = self.beta2 * self.sy + (1 - self.beta2) * dy ** 2

                    m_hat_x = self.mx / (1 - self.beta1 ** t)
                    m_hat_y = self.my / (1 - self.beta1 ** t)
                    s_hat_x = self.sx / (1 - self.beta2 ** t)
                    s_hat_y = self.sy / (1 - self.beta2 ** t)

                    x -= (self.lr * m_hat_x) / (np.sqrt(s_hat_x) + self.epsilon)
                    y -= (self.lr * m_hat_y) / (np.sqrt(s_hat_y) + self.epsilon)

                x = np.clip(x, -1e8, 1e8)
                y = np.clip(y, -1e8, 1e8)
                self.path.append((x, y))

            return self.path
        except Exception as e:
            print(f"Error in 2D gradient descent: {e}")
            return []
