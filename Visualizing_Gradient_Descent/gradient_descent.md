# 📘 Phase 1: Understanding Gradient Descent

---

## 🎯 What is Gradient Descent?

Gradient Descent is a **first-order iterative optimization algorithm** used to find the **minimum of a function**.

In machine learning, it’s commonly used to **minimize the loss function** by updating model parameters in the direction that most reduces the loss — the **negative of the gradient**.

---

## ⚙️ The Update Rule (1D)

Given a function `f(x)`, the gradient descent update rule is:

$$
x_{\text{new}} = x_{\text{old}} - \alpha \cdot \frac{df}{dx}
$$

Where:
- $ x $ is the current position
- $ \alpha $ is the **learning rate**
- $ \frac{df}{dx} $ is the derivative (slope) at point $ x $

---

## 📉 Example: `f(x) = x²`

To minimize this function using gradient descent:

### 🔢 Step-by-Step Derivation

$$
\frac{df}{dx} = 2x
$$

So, the update rule becomes:

$$
x_{\text{new}} = x - \alpha \cdot 2x = x(1 - 2\alpha)
$$

> 🔁 This leads to exponential decay towards the minimum at 0.

---

## 📊 Image Prompt

**Prompt**:  
Plot the function `f(x) = x²` with a red dot showing descent steps from `x = 4 → 0`. Annotate each step: "Step 0", "Step 1", ..., "Converged".

---

## ⚠️ Why Learning Rate Matters

### 🎯 Good Learning Rate

A good α is small enough to converge smoothly and large enough to converge fast.

**Prompt**:  
Overlay 3 descent paths with:
- α = 0.01 → slow convergence  
- α = 0.1 → good convergence  
- α = 1.2 → diverges

Use labeled points and arrows.

---

## 🔄 Extending to 2D Functions

For a function $ f(x, y) = x^2 + y^2 $, the **gradient vector** is:

$$
\nabla f = \left[ \frac{∂f}{∂x}, \frac{∂f}{∂y} \right] = [2x, 2y]
$$

Update rule becomes:

$$
\vec{x}_{\text{new}} = \vec{x}_{\text{old}} - \alpha \cdot \nabla f
$$

---

### 🧮 2D Descent Example

Start at point $(x = 2, y = 2)$, learning rate $ \alpha = 0.1 $

1. $ \nabla f = (4, 4) $
2. New point = $ (2 - 0.4, 2 - 0.4) = (1.6, 1.6) $
3. Continue until convergence at $ (0, 0) $

---

## 🌄 Image Prompt

**Prompt**:  
A 3D surface plot of `f(x, y) = x² + y²` with a red dot descending from `(2,2)` to `(0,0)` on the bowl. Show arrows for gradients.

---

## 🧠 Intuition Summary

- Always **move in the negative gradient direction** (steepest descent).
- Learning rate controls **step size**.
- Too large → diverges. Too small → slow.
- Requires function to be **differentiable**.

---

## ✅ Recap: Key Terms

| Term           | Description                                 |
|----------------|---------------------------------------------|
| Gradient       | Direction of steepest ascent                |
| Learning Rate  | Step size in parameter space                |
| Convergence    | Reaching (local/global) minimum             |
| Loss Function  | Objective to minimize in ML                 |
| Derivative     | Slope of function at a point                |
