# This is the same solution as used on ode00.py, but now includes the
# 2nd derivative.

from math import exp

# Define the analytical solution.
def ya(x):
    return 1 + exp(-x**2 / 2)

# Define the 1st analytical derivative.
def dya_dx(x):
    return -x * exp(-x**2 / 2)

# Define the 2nd analytical derivative.
def d2ya_dx2(x):
    return (x**2 - 1) * exp(-x**2 / 2)

# Define the differential equation.
def F(x, y):
    return (y - 1) * (x**2 - 1)

# Define the y-partial derivative of the differential equation.
def dF_dy(x, y):
    return x**2 - 1

# Define the 2nd y-partial derivative of the differential equation.
def d2F_dy2(x, y):
    return 0

# Boundary conditions
xmin = 0
xmax = 1
ymin = ya(xmin)   # 2
ymax = ya(xmax)   # 1 + 1/sqrt(e) = 1.60653...
