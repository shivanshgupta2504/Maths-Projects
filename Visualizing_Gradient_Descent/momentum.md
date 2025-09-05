# ⚡ Optimizer: Momentum

---

## 🎯 What is Momentum?

Momentum is a variant of gradient descent that **accelerates convergence** by using **velocity** (past gradients) to smooth updates.

It adds an **exponential moving average** of past gradients, helping to escape small local minima or oscillating paths.

---

## 🔁 Update Rule

Let $ v_t $ be the **velocity** at step t:

$$
v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot \nabla f(x_t) \\
x_{t+1} = x_t - \alpha \cdot v_t
$$

Where:
- $ \beta $ : momentum coefficient (usually 0.9)
- $ \alpha $ : learning rate

---

## 🧠 Why Use It?

- Helps in **flat regions** or **ravines**
- Reduces **oscillations**
- Speeds up descent in consistent directions

---

## 📊 Visualization Prompt

**Prompt**:  
Show function `f(x) = sin(x)` with:
- Normal GD oscillating
- Momentum optimizer converging faster with smooth path

---

## 🔧 Hyperparameter

| Parameter | Range  | Default |
|-----------|--------|---------|
| β (beta)  | 0.1–0.99 | 0.9   |

---

## ✅ Summary

| Feature       | Description                    |
|---------------|--------------------------------|
| Smoothness    | High                           |
| Convergence   | Faster on flat terrains        |
| Drawbacks     | May overshoot if β is too high |
