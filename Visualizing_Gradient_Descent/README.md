# 📉 Visualizing Gradient Descent in Action

An interactive Streamlit app that **visually demonstrates the working of Gradient Descent** on 1D and 2D mathematical functions. Great for anyone learning Machine Learning, Calculus, or Optimization!

---

## 🚀 Features (Phase 1–4 Complete)

### ✅ Phase 1: Basic 1D Gradient Descent
- Select from predefined **1D functions**: `x²`, `sin(x)`, `|x|`, `e^x`, etc.
- Adjustable **learning rate**, **steps**, and **initial point**
- **Live plot** of `f(x)` with descent path
- **Loss vs Steps** line graph

### ✅ Phase 2: UI Enhancements & State Management
- Two-way synced sliders + number input (for learning rate, x₀, steps)
- 🔁 **Reset to default values** instantly
- Modular structure with reusable utilities
- `st.session_state` used to maintain UI consistency

### ✅ Phase 3: Function Diversity & Side-by-Side Visuals
- Support for **multiple convex and non-convex functions**
- LaTeX-based **formula display** in sidebar
- Displays **final result** (minimum x and f(x))
- Adds **stationary points** for visual context
- ❗ Warning for unstable/invalid learning rates
- **Side-by-side layout**: Left = `f(x)` plot, Right = Loss vs Steps

### ✅ Phase 4: 2D Mode and Surface Visualization
- Toggle between **1D and 2D modes**
- Functions like `f(x, y) = x² + y²`, `x² - y²`, `x·sin(y)`
- 3D surface plots using `matplotlib`
- Step-by-step descent path overlayed on the 3D surface
- Two-way input sliders for `x₀`, `y₀`, learning rate, and steps
- Final `(x, y)` and `f(x, y)` output shown
- Warnings for divergent or non-converging functions (like saddle points)
- Full **feature parity** with 1D version (loss table, reset, etc.)

---

## 📚 Concepts Demonstrated

- Gradient Descent (1D and 2D)
- Derivatives and Partial Derivatives
- Learning Rate Tuning
- Function Minimization & Optimization
- Surface Topology (Convex, Non-convex, Saddle)
- Local Minima vs Global Minima

---

## 📊 Tech Stack

| Layer        | Tool / Library         |
|--------------|------------------------|
| GUI          | Streamlit              |
| Plots        | Matplotlib (3D, 2D)    |
| Animation    | FuncAnimation, Pillow  |
| Language     | Python 3.10+           |

---

## 🖥️ Installation

```bash
git clone https://github.com/yourusername/gradient-descent-visualizer.git
cd gradient-descent-visualizer
pip install -r requirements.txt
streamlit run main.py
