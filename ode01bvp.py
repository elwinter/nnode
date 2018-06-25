# Sample 2nd-order ODE BVP with Dirichlet BC

# A reasonable solution can be found using the following settings:
# eta=0.001

# The equation is defined on the domain [0,1], with the Dirichlet
# boundary conditions defined at x=0 and x=1.

# Define the original differential equation, assumed to be in the form
# G(x,y,dy/dx) = d2y_dx2 - dy_dx + 2x - 1 = 0
# Solution is y(x) = x**2 + x + 1
def Gf(x, y, dy_dx, d2y_dx2):
    return d2y_dx2 - dy_dx + 2*x - 1

# Boundary conditions at 0 and 1
bc0 = 1
bc1 = 3

# Derivatives of ODE

def dG_dyf(x, y, dy_dx, d2y_dx2):
    return 0

def dG_dydxf(x, y, dy_dx, d2y_dx2):
    return -1

def dG_d2ydx2f(x, y, dy_dx, d2y_dx2):
    return 1

# Define the analytical solution.
def yaf(x):
    return x**2 + x + 1

# Define the 1st analytical derivative.
def dya_dxf(x):
    return 2*x + 1

# Define the 2nd analytical derivative.
def d2ya_dx2f(x):
    return 2
