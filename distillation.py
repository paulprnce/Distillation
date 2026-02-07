import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import fsolve

st.set_page_config(page_title="McCabe–Thiele Simulator", layout="wide")

st.title("🧪 McCabe–Thiele Distillation Column Simulator")

st.sidebar.header("🔧 Input Parameters")

# -------- Inputs --------
xF = st.sidebar.number_input("Feed composition (xF)", 0.0, 1.0, 0.4)
xD = st.sidebar.number_input("Distillate composition (xD)", 0.0, 1.0, 0.95)
xB = st.sidebar.number_input("Bottom composition (xB)", 0.0, 1.0, 0.05)

alpha = st.sidebar.number_input("Relative volatility (α)", 1.1, 10.0, 2.4)
q = st.sidebar.number_input("Feed quality (q)", 0.0, 2.0, 1.0)

efficiency = st.sidebar.slider("Murphree Efficiency", 0.4, 1.0, 0.7)

calculate = st.sidebar.button("🚀 Run Simulation")

# -------- Equilibrium --------
def y_equilibrium(x, a):
    return (a * x) / (1 + (a - 1) * x)

# -------- Main Calculation --------
if calculate:

    x = np.linspace(0, 1, 500)
    y_eq = y_equilibrium(x, alpha)

    # ---- q-line ----
    def q_line(x):
        return (q / (q - 1)) * x - (xF / (q - 1))

    # ---- Find pinch point ----
    if q == 1:
        x_int = xF
        y_int = y_equilibrium(x_int, alpha)
    else:
        x_int = fsolve(lambda xx: q_line(xx) - y_equilibrium(xx, alpha), 0.5)[0]
        y_int = y_equilibrium(x_int, alpha)

    # ---- Minimum reflux ratio ----
    Rmin = (xD - y_int) / (y_int - x_int)

    # ---- Actual reflux ratio ----
    R = 1.5 * Rmin

    # ---- Operating lines ----
    def rectifying_line(x):
        return (R / (R + 1)) * x + xD / (R + 1)

    def stripping_line(x):
        return ((y_int - xB) / (x_int - xB)) * (x - xB) + xB

    # -------- Tray stepping --------
    x_curr = xD
    y_curr = xD
    stages = 0

    x_steps = []
    y_steps = []

    while x_curr > xB:
        # Horizontal to equilibrium
        x_eq = fsolve(lambda xx: y_equilibrium(xx, alpha) - y_curr, x_curr)[0]

        x_steps += [x_curr, x_eq]
        y_steps += [y_curr, y_curr]

        # Vertical to operating line
        if x_eq >= x_int:
            y_next = rectifying_line(x_eq)
        else:
            y_next = stripping_line(x_eq)

        x_steps += [x_eq, x_eq]
        y_steps += [y_curr, y_next]

        x_curr = x_eq
        y_curr = y_next
        stages += 1

        if stages > 60:   # safety break
            break

    theoretical_stages = stages
    actual_trays = theoretical_stages / efficiency

    # -------- OUTPUT --------
    st.markdown("## 📊 Results Summary")
    with st.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("Minimum Reflux Ratio (Rmin)", f"{Rmin:.3f}")
        col2.metric("Operating Reflux Ratio (R)", f"{R:.3f}")
        col3.metric("Theoretical Stages", theoretical_stages)

    st.metric("Actual Trays (with Efficiency)", f"{actual_trays:.1f}")

    # -------- PLOT --------
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(x, y_eq, label="Equilibrium Curve")
    ax.plot(x, x, "--", label="45° Line")

    ax.plot(x, rectifying_line(x), label="Rectifying Line")
    ax.plot(x, stripping_line(x), label="Stripping Line")

    if q == 1:
        ax.axvline(x=xF, linestyle="--", label="q-line (vertical)")
    else:
        ax.plot(x, q_line(x), label="q-line")

    # ---- Plot steps ----
    ax.plot(x_steps, y_steps, color="black", linewidth=1.5, label="Stages")

    ax.set_xlabel("x (liquid composition)")
    ax.set_ylabel("y (vapour composition)")
    ax.legend()
    ax.grid(True)
    st.markdown("## 📈 McCabe–Thiele Diagram")
    st.pyplot(fig)

else:
    st.info("👈 Enter parameters and click **Run Simulation**")


with st.expander("📘 Theory Behind McCabe–Thiele"):
    st.write("""
    McCabe–Thiele Method is a graphical technique used to design and
    analyze distillation columns for binary mixtures. It visually represents
    the equilibrium between liquid and vapor phases and helps estimate the number
    of theoretical stages required to achieve a desired separation. By plotting
    operating lines along with the vapor–liquid equilibrium curve, the method shows 
    how changes in reflux ratio, feed condition, and composition affect column performance.
    This approach is widely used in industry and academia because it offers clear physical insight
    into how a distillation column operates.
    """)