# 📘 Phase 1: Understanding Gradient Descent

---

## 🎯 What is Gradient Descent?

Gradient Descent is a **first-order optimization algorithm** used to find the **minimum of a function**.

In machine learning, it’s used to **minimize a loss function** by updating weights in the direction that reduces error the fastest — that direction is the **negative of the gradient**.

---

## ⚙️ The Update Rule (1D)

If you have a function `f(x)`, the update rule is:

$$
x_{\text{new}} = x_{\text{old}} - \alpha \cdot \frac{df}{dx}
$$

Where:
- $(x)$ is the current point
- $(\alpha)$ is the **learning rate**
- $\frac{df}{dx}$ is the derivative at point \( x \)

---

## 📉 Example: `f(x) = x²`

Let’s minimize this simple function using gradient descent.

### 🔢 Step-by-step:

$$
\frac{df}{dx} = 2x
$$

Update rule becomes:

$$
x_{\text{new}} = x - \alpha \cdot 2x = x(1 - 2\alpha)
$$

> 🔁 This recursive formula gives exponential decay towards 0.

---

## 📊 Plot (Image Prompt)

**Image Prompt**:  
Plot the function `f(x) = x²` with a red ball showing gradient descent steps from x=4 to x=0. Label each step as "Step 0", "Step 1", etc.

---

## ⚠️ Learning Rate (`α`) Matters

### 🎯 Good Learning Rate

Small enough to converge slowly but safely.

**Image Prompt**:  
Show 3 descent paths on `f(x) = x²` with α = 0.01 (slow), 0.1 (fast), 1.2 (diverges). Use dots/lines to show steps.

---

## 🔄 What if it's a 2D Function?

For `f(x, y) = x² + y²`, we compute the **gradient vector**:

$$
\nabla f = \left[ \frac{∂f}{∂x}, \frac{∂f}{∂y} \right] = [2x, 2y]
$$

Update rule:

$$
\vec{x}_{\text{new}} = \vec{x}_{\text{old}} - \alpha \cdot \nabla f
$$

---

### 🔢 2D Example:

Start at (x=2, y=2), α = 0.1

1. Gradient = (4, 4)  
2. New point = (2 - 0.4, 2 - 0.4) = (1.6, 1.6)  
3. Repeat until (0, 0)

---

## 🌄 2D Visualization (Image Prompt)

**Image Prompt**:  
A 3D surface plot of `f(x, y) = x² + y²` with a red dot starting at (2,2), showing its path downhill toward (0,0). Arrows should represent the gradient vector at each step.

---

## 🧠 Intuition Summary

- You’re always **moving downhill**, but **not too fast**.
- Learning rate too high? 🚨 You’ll jump over the minimum.
- Learning rate too low? 🐢 You’ll take forever.
- Works for any **differentiable function**!

---

## ✅ Recap: Key Terms

| Term | Meaning |
|------|---------|
| Gradient | Direction of steepest ascent |
| Learning Rate | Step size for updates |
| Convergence | Reaching minimum (or close) |
| Loss Function | What you want to minimize in ML |
| Derivative | Slope of the curve at a point |

---

