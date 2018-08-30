# This ODE is Problem 2 in Lagaris et al.

from math import cos, exp, sin

# The equation is defined on the domain [0,1], with the initial
# condition defined at x=0.

# The analytical form of the equation is:
# G(x,y,dy/dx) = dy/dx + y/5 - exp(-x/5)*cos(x)
#              = 0
def Gf(x, y, dy_dx):
    return dy_dx + y/5 - exp(-x/5)*cos(x)

# Initial condition
ic = 0

# Derivatives of ODE

def dG_dyf(x, y, dy_dx):
    return 1/5

def dG_dydxf(x, y, dy_dx):
    return 1

# Define the analytical solution.
def yaf(x):
    return exp(-x/5)*sin(x)

# Define the 1st analytical derivative.
def dya_dxf(x):
    return 1/5*exp(-x/5)*(5*cos(x) - sin(x))
