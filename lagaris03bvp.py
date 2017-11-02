# This ODE is Problem 3 in Lagaris et al (BVP version).

# A reasonable solution can be found using default parameter values.

# BC are set so that xmin is always 0, and ymin is f(0).

from math import exp, sin, cos

# Define the analytical solution.
def ya(x):
    return exp(-x / 5) * sin(x)

# Define the 1st analytical derivative.
def dya_dx(x):
    return (
        1 / 5 * exp(-x / 5) * (5 * cos(x) - sin(x))
    )

# Define the 2nd analytical derivative.
def d2ya_dx2(x):
    return (
        -2 / 25 * exp(-x / 5) * (5 * cos(x) + 12 * sin(x))
    )

# Define the differential equation.
def F(x, y, dy_dx):
    return (
        -1 / 5 * exp(-x / 5) * cos(x) - 1 / 5 * dy_dx - y
    )

# Define the y-partial derivative of the differential equation.
def dF_dy(x, y, dy_dx):
    return -1

# Define the 2nd y-partial derivative of the differential equation.
def d2F_dy2(x, y, dy_dx):
    return 0

# Boundary conditions
xmin = 0
xmax = 1
ymin = ya(xmin)   # 0
ymax = ya(xmax)   # sin(1) e^(-1 / 5) = 0.688938...
