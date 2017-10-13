from math import exp

# Boundary conditions. Only ymin or ymax can be valid, not both.
xmin = 0
xmax = 1
ymin = 2
ymax = None

# Define the analytical solution.
def yanal(x):
    return 1 + exp(-x**2 / 2)

# Define the original differential equation:
# dy/dx + x*y = x  ->  dy/dx = x*(1 - y) = F(x,y)
def F(x, y):
    return x * (1 - y)

# Define the y-partial derivative of the differential equation.
def dF_dy(x, y):
    return -x

# Define the 2nd y-partial derivative of the differential equation.
def d2F_dy2(x, y):
    return 0
