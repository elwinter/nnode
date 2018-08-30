# Sample 1st-order ODE IVP

# The equation is defined on the domain [0,1], with the initial
# condition defined at x=0.

# The analytical form of the equation is:
# G(x,y,dy/dx) = x - y = 0
def Gf(x, y, dy_dx):
    return x - y

# Initial condition
ic = 0

# Derivatives of ODE

def dG_dyf(x, y, dy_dx):
    return -1

def dG_dydxf(x, y, dy_dx):
    return 0

# Define the analytical solution.
def yaf(x):
    return x

# Define the 1st analytical derivative.
def dya_dxf(x):
    return 1
