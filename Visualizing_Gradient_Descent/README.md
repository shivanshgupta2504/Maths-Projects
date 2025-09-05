# 📉 Visualizing Gradient Descent in Action

An interactive **Streamlit app** that visually demonstrates the **working of Gradient Descent** (and its variants) on 1D and 2D mathematical functions. Designed for learners of **Machine Learning**, **Optimization**, or **Calculus**.

---

## 🚀 Features Completed (Phase 1–5)

### ✅ Phase 1: Basic 1D Gradient Descent
- Choose from multiple **predefined 1D functions**: `x²`, `sin(x)`, `|x|`, `e^x`, etc.
- Adjustable **learning rate**, **number of steps**, and **initial point (x₀)**
- 📈 **Function plot with descent path**
- 📉 **Loss vs Steps** graph to track convergence
- 🔣 Real-time update of minimum found

---

### ✅ Phase 2: UI Enhancements & State Management
- 🔁 Two-way synced **sliders + number input** for all values
- 🧹 **Reset button** to restore defaults for all inputs
- Clean modular layout using **`st.session_state`** for consistent behavior
- Utility abstraction for input handling (e.g., `slider_input()`)

---

### ✅ Phase 3: Function Diversity & Visualization Polish
- ✅ Support for **convex, non-convex, flat, and unstable** functions
- ➕ Live **LaTeX formula display** in sidebar
- 🧠 Final results (xₘᵢₙ and f(xₘᵢₙ)) shown with annotations
- ⚠️ Warnings for unstable learning rates (e.g., divergent sin/cubic)
- 📊 **Side-by-side layout**:
  - Left: Function descent plot
  - Right: Loss vs Steps graph

---

### ✅ Phase 4: 2D Mode and Surface/Contour Visualization
- 🔀 Toggle between **1D and 2D modes**
- 🌄 2D function support: `x² + y²`, `x² - y²`, `x·sin(y)`, etc.
- 🟦 3D surface plots with step-wise overlay
- 🔘 Sliders for `x₀`, `y₀`, learning rate, and steps
- 📌 Output shows `(x, y)` and `f(x, y)`
- 🔥 Warning messages for **non-converging** behavior (like saddle points)
- 🧩 Full **feature parity** with 1D version

---

### ✅ Phase 5: Animations & Optimizer Comparisons
- 🌀 Toggle to **enable/disable animation**
- 🎞️ **1D animated descent path** (GIF via Matplotlib + Pillow)
- 🌐 **2D Contour path animation** of optimization journey
- 🧠 Select from **4 optimizers**:
  - Gradient Descent (Vanilla)
  - Momentum
  - RMSprop
  - Adam
- ⚙️ Optimizer-specific sliders:
  - Momentum β
  - RMSprop: decay rate (ρ), epsilon (ε)
  - Adam: β₁, β₂, epsilon (ε)
- 🎛️ All hyperparameters adjustable via sliders (no number boxes)
- 🎯 Animations run in **sync** for both descent and loss graphs
- ✅ Modular engine design: logic separated into `gd_engine.py`, supports multiple optimizers

---

## 🧠 Concepts Demonstrated

- Gradient Descent Optimization (1D & 2D)
- Loss vs Epoch (Convergence Graph)
- Effect of Learning Rate on Descent Stability
- Role of Momentum, RMSprop, and Adam optimizers
- Local Minima vs Global Minima
- Saddle Point and Non-Convex Behavior
- Visualization of surface topology and optimization trajectory

---

## ⚙️ Tech Stack

| Layer        | Tool / Library         |
|--------------|------------------------|
| GUI          | Streamlit              |
| Plots        | Matplotlib (3D, 2D)    |
| Animation    | FuncAnimation, Pillow  |
| Core Logic   | Python Classes & OOP   |
| Version      | Python 3.10+           |

---

## 🖥️ Installation

```bash
git clone https://github.com/shivanshgupta2504/Maths-Projects.git
pip install -r requirements.txt
cd .\Visualizing_Gradient_Descent\
streamlit run main.py
