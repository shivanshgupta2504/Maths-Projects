# -----------------------------
# ✅ main.py
# -----------------------------
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gd_engine import Function1D, GradientDescent, GradientDescent2D
from functions import FUNCTIONS
from mpl_toolkits.mplot3d import Axes3D
from functions_2d import FUNCTIONS_2D
from matplotlib.animation import FuncAnimation
import io
import tempfile

# --- Helper: Two-way sync input + slider ---
def slider_input(
    label: str,
    slider_range: tuple[float, float],
    step: float,
    format_str: str,
    key_prefix: str,
    default: float
) -> float:
    """
    A cleaner one-way slider input for streamlined UI.
    """
    min_val, max_val = slider_range
    slider_key = f"{key_prefix}_slider"

    if slider_key not in st.session_state:
        st.session_state[slider_key] = default

    return st.sidebar.slider(
        f"{label}", min_value=min_val, max_value=max_val,
        value=st.session_state[slider_key], step=step,
        format=format_str, key=slider_key
    )

def plot_3d_descent(func, path: list[tuple[float, float]]) -> plt.Figure:
    """
    Plot a 3D surface of the function and the descent path.
    """
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    # Surface grid
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    Z = func.evaluate(X, Y)

    # Plot surface
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8, edgecolor='none')

    # Plot Path
    xs, ys = zip(*path)
    zs = [func.evaluate(x, y) for x, y in path]
    ax.plot(xs, ys, zs, color='red', marker='o', label='Descent Path')

    # Style
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel("f(x, y)")
    if func.name == "Saddle":
        ax.set_zlim(-50, 50)  # Prevents huge distortion
    ax.set_title('3D Gradient Descent Path')
    ax.view_init(elev=35, azim=45)
    ax.legend()

    return fig

# --- Animation Function for 1D Functions ---
def plot_combined_animation(x_vals: list[float], y_vals: list[float], function: Function1D):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # --- Left Plot: Descent Path
    x_range = np.linspace(-10, 10, 400)
    ax1.plot(x_range, function.evaluate(x_range), color='blue', label='Function')
    path_line, = ax1.plot([], [], 'ro-', label="Descent", markersize=5)
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.set_title("Descent Path")
    ax1.grid(True)

    # --- Right Plot: Loss Curve
    loss_line, = ax2.plot([], [], 'bo-', label="Loss")
    ax2.set_xlim(0, len(y_vals)-1)
    ax2.set_ylim(min(y_vals)-1, max(y_vals)+1)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("f(x)")
    ax2.set_title("Loss vs Steps")
    ax2.grid(True)

    def init():
        path_line.set_data([], [])
        loss_line.set_data([], [])
        return path_line, loss_line

    def update(frame):
        path_line.set_data(x_vals[:frame+1], y_vals[:frame+1])
        loss_line.set_data(list(range(frame+1)), y_vals[:frame+1])
        return path_line, loss_line

    ani = FuncAnimation(fig, update, frames=len(x_vals), init_func=init,
                        interval=300, blit=True, repeat=False)

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
        ani.save(tmpfile.name, writer='pillow', fps=2, dpi=100)
        tmpfile.seek(0)
        gif_bytes = tmpfile.read()

    st.image(gif_bytes, caption="📽️ Animated Gradient Descent & Loss Curve")

# --- Animation Function for 2D Contour Descent ---
def plot_animated_contour_2d(func, path):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Grid
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    Z = func.evaluate(X, Y)

    # Contour Lines
    contour = ax.contour(X, Y, Z, levels=30, cmap="viridis")
    ax.clabel(contour, inline=True, fontsize=8)

    # Path Setup
    x_vals, y_vals = zip(*path)
    line, = ax.plot([], [], "ro-", label="Descent Path", lw=2)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Gradient Descent on Contour Plot")
    ax.legend()
    ax.grid(True)

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        line.set_data(x_vals[:frame+1], y_vals[:frame+1])
        return (line,)

    ani = FuncAnimation(fig, update, frames=len(x_vals), init_func=init, interval=300, blit=True, repeat=False)

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
        ani.save(tmpfile.name, writer='pillow', fps=2, dpi=100)
        tmpfile.seek(0)
        gif_bytes = tmpfile.read()

    st.image(gif_bytes, caption="📽️ 2D Gradient Descent on Contour Plot")

# --- Streamlit Config ---
st.set_page_config(page_title="Gradient Descent Visualizer", layout="centered")
st.title("📉 Visualizing Gradient Descent in Action")

# --- Mode Switch (1D vs 2D) ---
mode = st.radio("Select Visualization Mode:", ["1D", "2D"], horizontal=True)

# -----------------------------
# 1D GRADIENT DESCENT BLOCK
# -----------------------------
if mode == "1D":
    st.subheader("📐 1D Gradient Descent")

    # --- Optimizer Selector ---
    optimizer_type = st.sidebar.selectbox(
        "🧠 Choose Optimizer",
        ["Gradient Descent", "Momentum", "RMSprop", "Adam"],
        key="optimizer_type"
    )

    # --- Function Selection ---
    function_names = list(FUNCTIONS.keys())
    selected_name = st.sidebar.selectbox("📐 Select 1D Function", function_names)
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

    learning_rate = slider_input("α", (0.001, 1.0), 0.001, "%.3f", "lr", DEFAULT_LR)
    initial_x = slider_input("x₀", (-10.0, 10.0), 0.1, "%.1f", "x0", DEFAULT_X)
    steps = int(slider_input("Steps", (1, 100), 1, "%d", "steps", DEFAULT_STEPS))

    # --- Optimizer Parameters (Dynamic Sidebar Sliders) ---
    optimizer_params = {}
    if optimizer_type == "Momentum":
        optimizer_params["momentum"] = slider_input(
            "Momentum (β)", (0.1, 0.99), 0.01, "%.2f", "momentum", 0.9
        )
    elif optimizer_type == "RMSprop":
        optimizer_params["decay_rate"] = slider_input(
            "Decay Rate (ρ)", (0.1, 0.99), 0.01, "%.2f", "decay_rate", 0.9
        )
        optimizer_params["epsilon"] = slider_input(
            "Epsilon (ε)", (1e-8, 1e-2), 1e-6, "%.0e", "rms_epsilon", 1e-5
        )
    elif optimizer_type == "Adam":
        optimizer_params["beta1"] = slider_input(
            "β₁ (Momentum)", (0.1, 0.999), 0.01, "%.3f", "beta1", 0.9
        )
        optimizer_params["beta2"] = slider_input(
            "β₂ (RMS)", (0.1, 0.999), 0.001, "%.3f", "beta2", 0.999
        )
        optimizer_params["epsilon"] = slider_input(
            "Epsilon (ε)", (1e-8, 1e-2), 1e-6, "%.0e", "adam_epsilon", 1e-5
        )

    # --- Function Class ---
    function = Function1D(
        func=selected_function.evaluate,
        derivative=selected_function.derivative
    )

    # --- Run Gradient Descent ---
    optimizer = GradientDescent(
        func=function,
        start_x=initial_x,
        lr=learning_rate,
        steps=steps,
        optimizer_type=optimizer_type,
        optimizer_params=optimizer_params
    )
    x_vals = optimizer.run()
    y_vals = [function.evaluate(x) for x in x_vals]

    # --- 1D Display ---
    st.markdown(f"### 📊 Selected Function: {selected_function.formula_latex}")
    animate_all = st.checkbox("🎞️ Animate Gradient Descent (Path + Loss)", key="animate_all")

    col1, col2 = st.columns(2)

    if animate_all:
        plot_combined_animation(x_vals, y_vals, function)
    else:
        with col1:
            st.markdown("#### 📉 Descent Path")
            # animate = st.checkbox("🎞️ Show Animated Descent", key="animate_1d")

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
            # animate_loss = st.checkbox("🎞️ Animate Loss Curve", key="animate_loss")
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
# -----------------------------
# 2D GRADIENT DESCENT BLOCK
# -----------------------------
else:
    st.subheader("🌄 2D Gradient Descent Visualization")

    # --- Optimizer Selector ---
    optimizer_type = st.sidebar.selectbox(
        "🧠 Choose Optimizer",
        ["Gradient Descent", "Momentum", "RMSprop", "Adam"],
        key="optimizer_type"
    )

    function_names_2d = list(FUNCTIONS_2D.keys())
    selected_name = st.sidebar.selectbox("📐 Select 2D Function", function_names_2d)
    selected_function = FUNCTIONS_2D[selected_name]

    DEFAULT_LR = selected_function.recommended_lr
    DEFAULT_X = 3.0
    DEFAULT_Y = 3.0
    DEFAULT_STEPS = 30

    if st.sidebar.button("🔁 Reset to Default"):
        st.session_state['2d_lr_val'] = DEFAULT_LR
        st.session_state['2d_x0_val'] = DEFAULT_X
        st.session_state['2d_y0_val'] = DEFAULT_Y
        st.session_state['2d_steps_val'] = DEFAULT_STEPS
        st.rerun()

    st.sidebar.markdown(selected_function.formula_latex)
    if not selected_function.has_minimum:
        st.sidebar.warning("⚠️ No global minimum — may diverge.")

    learning_rate = slider_input("α", (0.001, 1.0), 0.001, "%.3f", "2d_lr", DEFAULT_LR)
    x0 = slider_input("x₀", (-10.0, 10.0), 0.1, "%.1f", "2d_x0", DEFAULT_X)
    y0 = slider_input("y₀", (-10.0, 10.0), 0.1, "%.1f", "2d_y0", DEFAULT_Y)
    steps = int(slider_input("Steps", (1, 100), 1, "%d", "2d_steps", DEFAULT_STEPS))

    # --- Optimizer Parameters (Dynamic Sidebar Sliders) ---
    optimizer_params = {}
    if optimizer_type == "Momentum":
        optimizer_params["momentum"] = slider_input(
            "Momentum (β)", (0.1, 0.99), 0.01, "%.2f", "momentum", 0.9
        )
    elif optimizer_type == "RMSprop":
        optimizer_params["decay_rate"] = slider_input(
            "Decay Rate (ρ)", (0.1, 0.99), 0.01, "%.2f", "decay_rate", 0.9
        )
        optimizer_params["epsilon"] = slider_input(
            "Epsilon (ε)", (1e-8, 1e-2), 1e-6, "%.0e", "rms_epsilon", 1e-5
        )
    elif optimizer_type == "Adam":
        optimizer_params["beta1"] = slider_input(
            "β₁ (Momentum)", (0.1, 0.999), 0.01, "%.3f", "beta1", 0.9
        )
        optimizer_params["beta2"] = slider_input(
            "β₂ (RMS)", (0.1, 0.999), 0.001, "%.3f", "beta2", 0.999
        )
        optimizer_params["epsilon"] = slider_input(
            "Epsilon (ε)", (1e-8, 1e-2), 1e-6, "%.0e", "adam_epsilon", 1e-5
        )

    optimizer = GradientDescent2D(
        func=selected_function,
        start=(x0, y0),
        lr=learning_rate,
        steps=steps,
        optimizer_type=optimizer_type,
        optimizer_params=optimizer_params
    )
    path = optimizer.run()

    st.markdown(f"### 📊 Selected Function: {selected_function.formula_latex}")
    fig3d = plot_3d_descent(selected_function, path)
    st.pyplot(fig3d)

    if st.checkbox("🎞️ Animate 2D Descent on Contour Plot"):
        plot_animated_contour_2d(selected_function, path)

    st.subheader("📌 Final Results")
    if path:
        final_x, final_y = path[-1]
        final_loss = selected_function.evaluate(final_x, final_y)
        st.markdown(f"**Final (x, y):** ({final_x:.6f}, {final_y:.6f})")
        st.markdown(f"**Final f(x, y):** {final_loss:.6f}")
        st.markdown(f"**Initial Loss:** {selected_function.evaluate(x0, y0):.6f}")
        dx, dy = selected_function.gradient(final_x, final_y)
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            st.success("✅ Gradient is small — likely converged.")
        else:
            st.warning("⚠️ Gradient still large — tune LR or steps.")
        if st.checkbox("🧾 Show Step Table"):
            df = pd.DataFrame({
                "Step": range(len(path)),
                "x": [p[0] for p in path],
                "y": [p[1] for p in path],
                "f(x,y)": [selected_function.evaluate(p[0], p[1]) for p in path]
            })
            st.dataframe(df)
    else:
        st.error("❌ Gradient Descent failed. Adjust parameters.")
