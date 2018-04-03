# This ODE is Problem 1 in Lagaris et al.

from math import exp

# A reasonable solution can be found using the following settings:
# 

# The equation is defined on the domain [0,1], with the boundary
# conditions defined at x=0.

# Define the original differential equation, assumed to be in the form:
# G(x,y,dy/dx) = dy/dx + (x + (1 + 3*x**2)/(1 + x + x**3))*y
#                - x**3 - 2*x - x**2*(1 + 3*x**2)/(1 + x + x**3)
#              = 0
# Solution is y(x) = exp(-x**2/2) / (1 + x * x**3) + x**2
def Gf(x, y, dy_dx):
    return (
        dy_dx + (x + (1 + 3*x**2)/(1 + x + x**3))*y
        - x**3 - 2*x - x**2*(1 + 3*x**2)/(1 + x + x**3)
    )

# Initial condition
ic = 1

# Derivatives of ODE

def dG_dyf(x, y, dy_dx):
    return x + (1 + 3*x**2)/(1 + x + x**3)

def dG_dydxf(x, y, dy_dx):
    return 1

# Define the analytical solution.
def yaf(x):
    return exp(-x**2/2)/(1 + x + x**3) + x**2

# Define the 1st analytical derivative.
def dya_dxf(x):
    return 2*x - exp(-x**2/2)*(1 + x + 4*x**2 + x**4)/(1 + x + x**3)**2
