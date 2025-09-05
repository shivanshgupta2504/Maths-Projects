# 🤖 Optimizer: Adam (Adaptive Moment Estimation)

---

## 🎯 What is Adam?

Adam combines the benefits of **Momentum** and **RMSprop**.

It maintains both:
- An exponentially decaying average of past gradients (Momentum)
- An exponentially decaying average of past squared gradients (RMSprop)

---

## 🔁 Update Rule

Let $ m_t $ be the 1st moment (mean), and $ v_t $ the 2nd moment (variance):

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \\
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$

Bias-corrected versions:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

Update:

$$
x_{t+1} = x_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
$$

---

## 🧠 Why Use It?

- Fast convergence
- Handles sparse gradients well
- Works well out-of-the-box

---

## 🔧 Hyperparameters

| Parameter    | Default Value |
|--------------|---------------|
| $ \beta_1 $  | 0.9           |
| $ \beta_2 $  | 0.999         |
| $ \epsilon $ | 1e-8          |

---

## 📊 Visualization Prompt

**Prompt**:  
Compare Adam vs GD vs RMSprop on a bumpy 2D landscape. Show Adam reaching min faster and smoother.

---

## ✅ Summary

| Feature       | Description                      |
|---------------|----------------------------------|
| Speed         | High                             |
| Stability     | Excellent                        |
| Drawbacks     | May overfit on noisy objectives  |
