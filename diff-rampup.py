# 1-D diffusion PDE

# A reasonable solution can be found using the following settings:
#

from math import exp, pi, sin

# The equation is defined on the domain [[0,1],[0,1]]. The profile starts flat
# at Y=0. The value at x=0 then linearly increases with time.

# The analytical form of the equation is:

# G(x,t,Y,delY,deldelY) = dY_dt - D*d2Y_dx2 = 0

# Diffusion coefficient
D = 0.01

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
    return 0

def dY0_dxf(x):
    return 0

# Boundary conditions for x at 0,1: Ramp up linearly in time to K.
K = 0.1
def f0f(t):
    return K*t

def f1f(t):
    return 0

# Boundary conditions for t=0, no BC at t=1
def g0f(x):
    return Y0f(x)

def g1f(x):
    return None

# Array of BC functions
bcf = ((f0f, g0f), (f1f, g1f))

# 1st t derivative of BC functions for x at x = 0, 1
def df0_dtf(t):
    return K

def df1_dtf(t):
    return 0

# 1st x derivative of BC functions for t at t = 0, 1
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
    return None

# Array of BC 2nd derivatives
bcd2f = ((d2f0_dt2f, d2g0_dx2f), (d2f1_dt2f, d2g1_dx2f))

# Analytical solution is the same as the starting profile.
def Yaf(xt):
    (x, t) = xt
    Ya = K*t*(1 - x)
    for k in range(1,101):
        Ya -= 2*K*(1 - exp(-pi**2*t*D*k**2))*sin(pi*k*x)/k**3/pi**3
    return Ya
