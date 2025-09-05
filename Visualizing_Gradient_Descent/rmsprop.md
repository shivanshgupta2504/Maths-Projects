# 📦 Optimizer: RMSprop

---

## 🎯 What is RMSprop?

RMSprop adapts the learning rate for each parameter by maintaining a **moving average of squared gradients**.

It **slows updates** for parameters with consistently large gradients and **speeds up** others.

---

## 🔁 Update Rule

1. Compute moving average of squared gradients:

$$
E[g^2]_t = \rho \cdot E[g^2]_{t-1} + (1 - \rho) \cdot (g_t)^2
$$

2. Update parameters:

$$
x_{t+1} = x_t - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t
$$

---

## 🧠 Why Use It?

- Handles **non-stationary objectives**
- Adapts to **different gradient magnitudes**
- Prevents exploding/vanishing gradients

---

## 📊 Visualization Prompt

**Prompt**:  
Show function `f(x) = e^x + sin(x)` with:
- Normal GD jumping wildly
- RMSprop stabilizing descent path

---

## 🔧 Hyperparameters

| Parameter    | Description            | Typical Value |
|--------------|------------------------|----------------|
| $ \rho $     | Decay rate             | 0.9            |
| $ \epsilon $ | Numerical stability     | 1e-8           |

---

## ✅ Summary

| Feature       | Description                     |
|---------------|---------------------------------|
| Smoothness    | Adaptive                        |
| Convergence   | Works well in non-convex space  |
| Drawbacks     | Slower on simple convex problems|
