# This ODE is the second example from the Lagaris paper.

# BC are set so that xmin is always 0, and ymin is f(0).

from math import cos, exp, sin

# Define the analytical solution (verified in Mathematica).
def ya(x):
    return exp(-x / 5) * sin(x)

# Define the 1st analytical derivative (verified in Mathematica).
def dya_dx(x):
    return 1 / 5 * exp(-x / 5) * (5 * cos(x) - sin(x))

# Define the original differential equation (Lagaris eq (28)).
def F(x, y):
    return exp(-x / 5) * cos(x) - y / 5

# Define the 1st y-partial derivative of the differential equation.
def dF_dy(x, y):
    return -1 / 5

# Define the 2nd y-partial derivative of the differential equation.
def d2F_dy2(x, y):
    return 0

# Boundary conditions
xmin = 0
xmax = 2
ymin = ya(xmin)   # 0
