from math import exp

# Define the analytical solution.
def yanal(x):
    return exp(-x**2 / 2) / (1 + x + x**3) + x**2

# Define the original differential equation:
# dy/dx + x*y = x  ->  dy/dx = x*(1 - y) = F(x,y)
def F(x, y):
    return (
        x**3 + 2 * x + x**2 * (1 + 3 * x**2) / (1 + x + x**3)
        - (x + (1 + 3 * x**2) / (1 + x + x**3)) * y
    )

# Define the y-partial derivative of the differential equation.
def dF_dy(x, y):
    return -(x + (1 + 3 * x**2) / (1 + x + x**3))

# Define the 2nd y-partial derivative of the differential equation.
def d2F_dy2(x, y):
    return 0

# Define the trial solution for this differential equation.
# Ref: Lagaris eq. (12), my equation (2)
def ytrial(x, N):
    A = 1
    return A + x * N

# Define the first trial derivative.
def dytrial_dx(x, N, Ng):
    return x * Ng + N
