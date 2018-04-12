# Sample 1st-order ODE IVP

# A reasonable solution can be found using the following settings:
# All defaults

# The equation is defined on the domain [0,1], with the initial
# condition defined at x=0.

# Define the original differential equation, assumed to be in the form
# G(x,y,dy/dx) = dy/dx - y + x**2 - 1 = 0
# Solution is y(x) = x**2 + 2*x + 1
def Gf(x, y, dy_dx):
    return dy_dx - y + x**2 - 1

# Initial condition
ic = 1

# Derivatives of ODE

def dG_dyf(x, y, dy_dx):
    return -1

def dG_dydxf(x, y, dy_dx):
    return 1

# Define the analytical solution.
def yaf(x):
    return x**2 + 2*x + 1

# Define the 1st analytical derivative.
def dya_dxf(x):
    return 2*x + 2
