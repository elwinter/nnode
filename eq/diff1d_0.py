# 1-D diffusion PDE

# The equation is defined on the domain [[0,1],[0,1]]. The initial
# profile is flat (Y(x,0)=0), and the BC at x=0,1 are fixed at
# Y(0,t)=Y(1,t)=0. The analytical solution is the same as the starting
# profile.

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

# (x,t) = (0,t)
def f0f(t):
    return 0

# (x,t) = (1,t)
def f1f(t):
    return 0

# (x,t) = (x,0)
def g0f(x):
    return 0

# (x,t) = (x,1) NOT USED
def g1f(x):
    return None

# Array of BC functions
bcf = ((f0f, g0f), (f1f, g1f))

# 1st t derivative of BC functions for x at x = 0,1
def df0_dtf(t):
    return 0

def df1_dtf(t):
    return 0

# 1st x derivative of BC functions for t at t = 0,1
def dg0_dxf(x):
    return 0

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
    return 0
