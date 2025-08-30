# 📉 Visualizing Gradient Descent in Action

An interactive web app that visually demonstrates how **Gradient Descent** works on different mathematical functions. Ideal for learners of Machine Learning, Calculus, or Optimization!

---

## 🚀 Features

- 📈 1D Function Visualizer (`f(x) = x²`, `sin(x)`, etc.)
- 🌄 2D Surface Visualizer (`f(x, y) = x² + y²`)
- 🎛️ Sliders to control:
  - Learning Rate
  - Number of Steps
  - Initial Point
- 📉 Live Loss vs Epoch Plot
- 📌 Supports convex and non-convex functions

---

## 📚 Concepts Covered

- Gradient Descent Algorithm  
- Derivatives and Gradients  
- Convergence and Learning Rate  
- Loss Minimization

---

## 🛠️ Tech Stack

- **Frontend/GUI**: [Streamlit](https://streamlit.io/)
- **Visualization**: Matplotlib / Plotly
- **Language**: Python 3.8+

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/gradient-descent-visualizer.git
cd gradient-descent-visualizer
pip install -r requirements.txt
streamlit run main.py
