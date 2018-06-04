# 1-D diffusion PDE

# A reasonable solution can be found using the following settings:
#

from math import exp, pi, sin

# The equation is defined on the domain [[0,1],[0,1]], fixed BC at
# x=0,1, and a triangular starting profile from boundary (Y=0 at
# x=0,1) to center (x=0.5,Y=0.5).

# The analytical form of the equation is:

# G(x,t,Y,delY,deldelY) = dY_dt - D*d2Y_dx2 = 0

# Diffusion coefficient
D = 1

# Define the original differential equation.
def Gf(xt, Y, delY, deldelY):
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    ((d2Y_dxdx, d2Y_dxdt), (d2Y_dtdx, d2Y_dtdt)) = deldelY
    return dY_dt - D*d2Y_dxdx

# Partial of PDE wrt Y
def dG_dYf(xt, Y, delY, deldelY):
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    ((d2Y_dxdx, d2Y_dxdt), (d2Y_dtdx, d2Y_dtdt)) = deldelY
    return 0

# Partials of PDE wrt del Y (partials wrt gradient of Y)
def dG_dY_dxf(xt, Y, delY, deldelY):
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    ((d2Y_dxdx, d2Y_dxdt), (d2Y_dtdx, d2Y_dtdt)) = deldelY
    return 0

def dG_dY_dtf(xt, Y, delY, deldelY):
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    ((d2Y_dxdx, d2Y_dxdt), (d2Y_dtdx, d2Y_dtdt)) = deldelY
    return 1

dG_ddelYf = (dG_dY_dxf, dG_dY_dtf)

# Partials of PDE wrt del del Y (partials wrt Hessian of Y)
def dG_d2Y_dxdxf(xt, Y, delY, deldelY):
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    ((d2Y_dxdx, d2Y_dxdt), (d2Y_dtdx, d2Y_dtdt)) = deldelY
    return -D

def dG_d2Y_dxdtf(xt, Y, delY, deldelY):
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    ((d2Y_dxdx, d2Y_dxdt), (d2Y_dtdx, d2Y_dtdt)) = deldelY
    return 0

def dG_d2Y_dtdxf(xt, Y, delY, deldelY):
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    ((d2Y_dxdx, d2Y_dxdt), (d2Y_dtdx, d2Y_dtdt)) = deldelY
    return 0

def dG_d2Y_dtdtf(xt, Y, delY, deldelY):
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    ((d2Y_dxdx, d2Y_dxdt), (d2Y_dtdx, d2Y_dtdt)) = deldelY
    return 0

dG_ddeldelYf = ((dG_d2Y_dxdxf, dG_d2Y_dxdtf),
                (dG_d2Y_dtdxf, dG_d2Y_dtdtf))

# Define the initial profile and its spatial derivative.
def Y0f(x):
    if x <= 0.5:
        return x
    else:
        return 1 - x

def dY0_dxf(x):
    if x <= 0.5:
        return 1
    else:
        return -1

# Boundary conditions for x at 0,1: clamped at x=0,1
def f0f(t):
    return Y0f(0)

def f1f(t):  # clamped at x=0,1
    return Y0f(1)

# Boundary conditions for t=0: triangular profile in x
def g0f(x):
    return Y0f(x)

def g1f(x): # No BC here
    return None

# Array of BC functions
bcf = ((f0f, g0f), (f1f, g1f))

# 1st derivative of BC functions for x at 0 and 1
def df0_dtf(t):
    return 0

def df1_dtf(t):
    return 0

# 1st derivative of BC functions for t at 0
def dg0_dxf(x):
    return dY0_dxf(x)

def dg1_dxf(x):
    return None

# Array of BC 1st derivatives
bcdf = ((df0_dtf, dg0_dxf), (df1_dtf, dg1_dxf))

# 2nd derivatives of BC functions for x at 0 and 1
def d2f0_dt2f(t):
    return 0

def d2f1_dt2f(t):
    return 0

# 2nd derivatives of BC functions for y at 0 and 1
def d2g0_dx2f(x):
    return 0

def d2g1_dx2f(x):
    return 0

# Array of BC 2nd derivatives
bcd2f = ((d2f0_dt2f, d2g0_dx2f), (d2f1_dt2f, d2g1_dx2f))

# Analytical solution is a Fourier series
def Yaf(xt):
    (x, t) = xt
    Ya = 0
    for k in range(1, 101):
        Ya += exp(-k**2*pi**2*D*t)*sin(k*pi/2)*sin(k*pi*x)/k**2
    Ya *= 4/pi**2
    return Ya
