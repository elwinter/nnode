from math import cos, exp, sin

# Boundary condition: y(b) = A
b = 0
A = 0

# Define the analytical solution.
def yanal(x):
    return exp(-x / 5) * sin(x)

# Define the original differential equation:
# dy/dx + x*y = x  ->  dy/dx = x*(1 - y) = F(x,y)
def F(x, y):
    return (
        exp(-x / 5) * cos(x) - y / 5
    )

# Define the y-partial derivative of the differential equation.
def dF_dy(x, y):
    return -1 / 5

# Define the 2nd y-partial derivative of the differential equation.
def d2F_dy2(x, y):
    return 0
