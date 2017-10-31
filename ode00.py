# Sample 1st-order ODE IVP

# A reasonable solution can be found using default settings.

# BC are set so that xmin is always 0, and ymin is f(0).

from math import exp

# Define the analytical solution.
def ya(x):
    return 1 + exp(-x**2 / 2)

# Define the 1st analytical derivative.
def dya_dx(x):
    return -x * exp(-x**2 / 2)

# Define the original differential equation:
def F(x, y):
    return x * (1 - y)

# Define the 1st y-partial derivative of the differential equation.
def dF_dy(x, y):
    return -x

# Define the 2nd y-partial derivative of the differential equation.
def d2F_dy2(x, y):
    return 0

# Boundary conditions
xmin = 0
xmax = 1
ymin = ya(xmin)   # 2
