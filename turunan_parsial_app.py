import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Aplikasi Turunan Parsial", layout="wide")

st.title("📊 Aplikasi Interaktif Turunan Parsial")
st.write("Masukkan fungsi biaya produksi dan titik evaluasi untuk menghitung turunan parsial serta menampilkan grafik.")

# Input user
f_str = st.text_input("Fungsi f(x, y):", "2*x**2 + 3*x*y + 4*y**2 + 10*x + 8*y")
x0 = st.number_input("x₀", value=2.0)
y0 = st.number_input("y₀", value=3.0)

x, y = sp.symbols('x y')
try:
    f = sp.sympify(f_str)

    # Turunan parsial
    dfdx = sp.diff(f, x)
    dfdy = sp.diff(f, y)

    # Evaluasi turunan di titik (x0, y0)
    dfdx_val = dfdx.evalf(subs={x: x0, y: y0})
    dfdy_val = dfdy.evalf(subs={x: x0, y: y0})
    f_val = f.evalf(subs={x: x0, y: y0})

    st.latex(r"f(x, y) = " + sp.latex(f))
    st.write("Turunan parsial terhadap x:", dfdx)
    st.write("Turunan parsial terhadap y:", dfdy)
    st.write(f"Nilai ∂f/∂x di ({x0}, {y0}): {dfdx_val}")
    st.write(f"Nilai ∂f/∂y di ({x0}, {y0}): {dfdy_val}")

    # Bidang singgung
    tangent_plane = f_val + dfdx_val * (x - x0) + dfdy_val * (y - y0)

    # Plotting
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    X_vals = np.linspace(x0 - 2, x0 + 2, 50)
    Y_vals = np.linspace(y0 - 2, y0 + 2, 50)
    X, Y = np.meshgrid(X_vals, Y_vals)

    f_lambd = sp.lambdify((x, y), f, "numpy")
    tangent_lambd = sp.lambdify((x, y), tangent_plane, "numpy")

    Z = f_lambd(X, Y)
    Z_tangent = tangent_lambd(X, Y)

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, label='Permukaan')
    ax.plot_surface(X, Y, Z_tangent, cmap='coolwarm', alpha=0.5)
    ax.scatter(x0, y0, f_val, color='red', s=50)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('Permukaan Fungsi dan Bidang Singgung')

    st.pyplot(fig)

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
