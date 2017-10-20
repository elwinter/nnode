# This ODE is the first example from the Lagaris paper.

# BC are set so that xmin is always 0, and ymin is f(0).

from math import exp

# Define the analytical solution (verified in Mathematica).
def ya(x):
    return exp(-x**2 / 2) / (1 + x + x**3) + x**2

# Define the 1st analytical derivative (verified in Mathematica).
def dya_dx(x):
    return (
        2 * x - exp(-x**2 / 2) * (1 + 3 * x**2) / (1 + x + x**3)**2
        - exp(-x**2 / 2) * x / (1 + x + x**3)
    )

# Define the original differential equation (Lagaris eq (27)).
def F(x, y):
    return (
        x**3 + 2 * x + x**2 * (1 + 3 * x**2) / (1 + x + x**3)
        - (x + (1 + 3 * x**2) / (1 + x + x**3)) * y
    )

# Define the 1st y-partial derivative of the differential equation.
def dF_dy(x, y):
    return -(x + (1 + 3 * x**2) / (1 + x + x**3))

# Define the 2nd y-partial derivative of the differential equation.
def d2F_dy2(x, y):
    return 0

# Boundary conditions
xmin = 0
xmax = 1
ymin = ya(xmin)   # 1
