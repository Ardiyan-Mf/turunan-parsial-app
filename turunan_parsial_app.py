# partial_derivative_app.py

import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Judul aplikasi
st.title("Aplikasi Turunan Parsial dan Visualisasi Bidang Singgung")

# Input fungsi dan titik evaluasi
st.markdown("## Masukkan Fungsi dan Titik Evaluasi")

x, y = sp.symbols('x y')
fungsi_input = st.text_input("Masukkan fungsi f(x, y):", "5*x**2 + 3*x*y + 2*y**2")

x0 = st.number_input("x₀ (titik evaluasi):", value=1.0)
y0 = st.number_input("y₀ (titik evaluasi):", value=2.0)

try:
    f = sp.sympify(fungsi_input)

    # Hitung turunan parsial
    dfdx = sp.diff(f, x)
    dfdy = sp.diff(f, y)

    dfdx_val = dfdx.subs({x: x0, y: y0})
    dfdy_val = dfdy.subs({x: x0, y: y0})

    st.markdown("## Hasil Turunan Parsial")
    st.latex(r"\frac{\partial f}{\partial x} = " + sp.latex(dfdx) + f" \\Rightarrow {dfdx_val}")
    st.latex(r"\frac{\partial f}{\partial y} = " + sp.latex(dfdy) + f" \\Rightarrow {dfdy_val}")

    # Konversi fungsi ke fungsi numpy
    f_np = sp.lambdify((x, y), f, "numpy")

    # Grafik 3D
    st.markdown("## Visualisasi Fungsi dan Bidang Singgung")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    X = np.linspace(x0 - 2, x0 + 2, 50)
    Y = np.linspace(y0 - 2, y0 + 2, 50)
    X, Y = np.meshgrid(X, Y)
    Z = f_np(X, Y)

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    # Bidang singgung
    z0 = f_np(x0, y0)
    Z_tangent = dfdx_val * (X - x0) + dfdy_val * (Y - y0) + z0
    ax.plot_surface(X, Y, Z_tangent, color='red', alpha=0.5)

    ax.scatter(x0, y0, z0, color='black', s=50)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.set_title("Permukaan Fungsi dan Bidang Singgung")

    st.pyplot(fig)

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
