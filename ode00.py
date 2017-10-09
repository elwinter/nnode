from math import exp

# x-value for boundary condition A (b = 0 means an IVP)
b = 0
A = 2

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

# Define the trial solution for this differential equation.
# Ref: Lagaris eq. (12), my equation (2)
def ytrial(x, N):
    return A + (x - b) * N

# Define the first trial derivative.
def dytrial_dx(x, N, Ng):
    return (x - b) * Ng + N
