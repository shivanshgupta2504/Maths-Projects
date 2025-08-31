import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from gd_engine import Function1D, GradientDescent

# --- Constants ---
DEFAULT_LR = 0.1
DEFAULT_X = 5.0
DEFAULT_STEPS = 20

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Gradient Descent Visualizer", layout="centered")
st.title("📉 Visualizing Gradient Descent in Action")
st.markdown("This app demonstrates how **Gradient Descent** works on the function **f(x) = x²**.")

# --- Helper: Two-way sync input + slider ---
def two_way_input(
    label: str,
    slider_range: tuple[float, float],
    step: float,
    format_str: str,
    key_prefix: str,
    default: float
) -> float:
    """
    Creates a synchronized number_input and slider that update each other.

    Args:
        label (str): Display label for both widgets.
        slider_range (tuple): (min, max) range.
        step (float): Step size.
        format_str (str): Format string (%.3f, etc.).
        key_prefix (str): Unique key prefix (e.g., 'lr', 'x0').
        default (float): Default value.

    Returns:
        float: The synchronized value.
    """
    min_val, max_val = slider_range
    state_key = f"{key_prefix}_val"
    slider_key = f"{key_prefix}_slider"
    input_key = f"{key_prefix}_input"

    if state_key not in st.session_state:
        st.session_state[state_key] = default

    # Number Input
    input_val = st.sidebar.number_input(
        f"Enter {label}", min_value=min_val, max_value=max_val,
        value=st.session_state[state_key], step=step, format=format_str,
        key=input_key
    )

    # Slider
    slider_val = st.sidebar.slider(
        f"Adjust {label}", min_value=min_val, max_value=max_val,
        value=st.session_state[state_key], step=step, key=slider_key
    )

    # Sync whichever changed
    if slider_val != st.session_state[state_key]:
        st.session_state[state_key] = slider_val
    elif input_val != st.session_state[state_key]:
        st.session_state[state_key] = input_val

    return st.session_state[state_key]

# --- Reset Button ---
if st.sidebar.button("🔁 Reset to Default"):
    st.session_state.lr_val = DEFAULT_LR
    st.session_state.x0_val = DEFAULT_X
    st.session_state.steps_val = DEFAULT_STEPS
    st.rerun()

# --- Sidebar Inputs ---
st.sidebar.header("🔧 Parameters")
st.sidebar.info("Use either the slider or the input box. Both stay in sync!")

st.sidebar.markdown("### Learning Rate (α)")
learning_rate = two_way_input("α", (0.001, 1.0), 0.001, "%.3f", "lr", DEFAULT_LR)

st.sidebar.markdown("### Initial Value (x₀)")
initial_x = two_way_input("x₀", (-10.0, 10.0), 0.1, "%.1f", "x0", DEFAULT_X)

st.sidebar.markdown("### Steps")
steps = int(two_way_input("Steps", (1, 100), 1, "%d", "steps", DEFAULT_STEPS))

# --- Function Definition ---
def f_x_squared(x: float) -> float:
    return x ** 2

def df_x_squared(x: float) -> float:
    return 2 * x

function = Function1D(func=f_x_squared, derivative=df_x_squared)

# --- Run Gradient Descent ---
optimizer = GradientDescent(func=function, start_x=initial_x, lr=learning_rate, steps=steps)
x_vals = optimizer.run()
y_vals = [function.evaluate(x) for x in x_vals]

# --- Plot Function and Descent Path ---
fig1, ax1 = plt.subplots()
x_range = np.linspace(-10, 10, 400)
ax1.plot(x_range, function.evaluate(x_range), label="$f(x) = x^2$", color='blue')
ax1.plot(x_vals, y_vals, 'ro-', label="Descent Path", markersize=5)
ax1.set_xlabel("x")
ax1.set_ylabel("f(x)")
ax1.set_title("Gradient Descent on $f(x) = x^2$")
ax1.legend()
ax1.grid(True)
fig1.tight_layout()
st.pyplot(fig1)

# --- Plot Loss vs Steps ---
fig2, ax2 = plt.subplots()
ax2.plot(range(len(y_vals)), y_vals, 'bo-', label="Loss")
ax2.set_xlabel("Step")
ax2.set_ylabel("f(x)")
ax2.set_title("Loss vs Steps")
ax2.grid(True)
fig2.tight_layout()
st.pyplot(fig2)

# --- Results Output ---
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
else:
    st.error("Gradient Descent failed. Please check your parameters.")
