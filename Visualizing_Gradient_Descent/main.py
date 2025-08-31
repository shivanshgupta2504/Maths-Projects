# -----------------------------
# ✅ main.py
# -----------------------------
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gd_engine import Function1D, GradientDescent
from functions import FUNCTIONS

# --- Helper: Two-way sync input + slider ---
def two_way_input(
    label: str,
    slider_range: tuple[float, float],
    step: float,
    format_str: str,
    key_prefix: str,
    default: float
) -> float:
    min_val, max_val = slider_range
    state_key = f"{key_prefix}_val"
    slider_key = f"{key_prefix}_slider"
    input_key = f"{key_prefix}_input"

    if state_key not in st.session_state:
        st.session_state[state_key] = default

    input_val = st.sidebar.number_input(
        f"Enter {label}", min_value=min_val, max_value=max_val,
        value=st.session_state[state_key], step=step, format=format_str,
        key=input_key
    )

    slider_val = st.sidebar.slider(
        f"Adjust {label}", min_value=min_val, max_value=max_val,
        value=st.session_state[state_key], step=step, key=slider_key
    )

    if slider_val != st.session_state[state_key]:
        st.session_state[state_key] = slider_val
    elif input_val != st.session_state[state_key]:
        st.session_state[state_key] = input_val

    return st.session_state[state_key]

# --- Streamlit Config ---
st.set_page_config(page_title="Gradient Descent Visualizer", layout="centered")
st.title("📉 Visualizing Gradient Descent in Action")

# --- Function Selection ---
function_names = list(FUNCTIONS.keys())
selected_name = st.sidebar.selectbox("📐 Select Function", function_names)
selected_function = FUNCTIONS[selected_name]

DEFAULT_LR = selected_function.recommended_lr
DEFAULT_X = 5.0
DEFAULT_STEPS = 20

# --- Reset Button ---
if st.sidebar.button("🔁 Reset to Default"):
    st.session_state.lr_val = DEFAULT_LR
    st.session_state.x0_val = DEFAULT_X
    st.session_state.steps_val = DEFAULT_STEPS
    st.rerun()

# --- Sidebar Inputs ---
st.sidebar.header("🔧 Parameters")
st.sidebar.markdown(selected_function.formula_latex)

if not selected_function.has_minimum:
    st.sidebar.warning("⚠️ This function has no global minimum — gradient descent may diverge.")

learning_rate = two_way_input("α", (0.001, 1.0), 0.001, "%.3f", "lr", DEFAULT_LR)
initial_x = two_way_input("x₀", (-10.0, 10.0), 0.1, "%.1f", "x0", DEFAULT_X)
steps = int(two_way_input("Steps", (1, 100), 1, "%d", "steps", DEFAULT_STEPS))

# --- Function Class ---
function = Function1D(
    func=selected_function.evaluate,
    derivative=selected_function.derivative
)

# --- Run Gradient Descent ---
optimizer = GradientDescent(func=function, start_x=initial_x, lr=learning_rate, steps=steps)
x_vals = optimizer.run()
y_vals = [function.evaluate(x) for x in x_vals]

# --- Main Display ---
st.markdown(f"### 📊 Selected Function: {selected_function.formula_latex}")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📉 Descent Path")
    fig1, ax1 = plt.subplots()
    x_range = np.linspace(-10, 10, 400)
    ax1.plot(x_range, function.evaluate(x_range), label=selected_function.formula_latex, color='blue')
    ax1.plot(x_vals, y_vals, 'ro-', label="Descent Path", markersize=5)

    # Stationary points
    for pt in getattr(selected_function, "stationary_points", []):
        ax1.axvline(x=pt, color='gray', linestyle='--', alpha=0.3)

    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.set_title(f"Gradient Descent on {selected_function.name}")
    ax1.legend()
    ax1.grid(True)
    fig1.tight_layout()
    st.pyplot(fig1)

with col2:
    st.markdown("#### 📈 Loss vs Steps")
    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(y_vals)), y_vals, 'bo-', label="Loss")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("f(x)")
    ax2.set_title("Loss vs Steps")
    ax2.grid(True)
    fig2.tight_layout()
    st.pyplot(fig2)

# --- Final Output ---
st.subheader("📌 Final Results")
if x_vals:
    st.markdown(f"**Final x:** {x_vals[-1]:.6f}")
    st.markdown(f"**Final f(x):** {y_vals[-1]:.6f}")
    st.markdown(f"**Initial Loss:** {y_vals[0]:.6f}")

    try:
        grad = function.gradient(x_vals[-1])
        if abs(grad) < 1e-3:
            st.success("✅ Gradient is small — likely converged.")
        else:
            st.warning("⚠️ Gradient still large — consider tuning learning rate or steps.")
    except Exception as e:
        st.error(f"Error during convergence check: {e}")

    if st.checkbox("🧾 Show Step-by-step Table"):
        data = pd.DataFrame({
            "Step": range(len(x_vals)),
            "x": x_vals,
            "f(x)": y_vals
        })
        st.dataframe(data)
else:
    st.error("Gradient Descent failed. Please check your parameters.")
