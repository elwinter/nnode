from math import sin, cos, pi

# Define the analytical solution.
def ya(x):
    return sin(x)

# Define the 1st analytical derivative.
def dya_dx(x):
    return cos(x)

# Define the 2nd analytical derivative.
def d2ya_dx2(x):
    return -sin(x)

# Define the differential equation.
def F(x, y):
    return -y

# Define the y-partial derivative of the differential equation.
def dF_dy(x, y):
    return -1

# Define the 2nd y-partial derivative of the differential equation.
def d2F_dy2(x, y):
    return 0

# Boundary conditions
xmin = 0
xmax = 2
ymin = ya(xmin)
ymax = ya(xmax)
