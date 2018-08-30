# Sample 1st-order ODE IVP

from math import sqrt, sin, cos

# The equation is defined on the domain [0,1], with the boundary
# conditions defined at x=0.

# The analytical form of the equation is:
# G(x,y,dy/dx) = dy/dx - cos(x) = 0
def Gf(x, y, dy_dx):
    return dy_dx - cos(x)

# Initial condition
ic = 0

# Derivatives of ODE

def dG_dyf(x, y, dy_dx):
    return 0

def dG_dydxf(x, y, dy_dx):
    return 1

# Define the analytical solution.
def yaf(x):
    return sin(x)

# Define the 1st analytical derivative.
def dya_dxf(x):
    return cos(x)
