# Sample 1st-order ODE IVP

from math import exp

# The equation is defined on the domain [0,1], with the initial
# condition defined at x=0.

# The analytical form of the equation is:
# G(x,y,dy/dx) = dy/dx - y + x**2 - 1 = 0
def Gf(x, y, dy_dx):
    return dy_dx - y + x**2 - 1

# Initial condition
ic = 0

# Derivatives of ODE

def dG_dyf(x, y, dy_dx):
    return -1

def dG_dydxf(x, y, dy_dx):
    return 1

# Define the analytical solution.
def yaf(x):
    return (x + 1)**2 - exp(x)

# Define the 1st analytical derivative.
def dya_dxf(x):
    return 2*(x + 1) - exp(x)
