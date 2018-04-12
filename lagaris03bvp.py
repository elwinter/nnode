# This ODE is Problem 3 in Lagaris et al (BVP version).

# A reasonable solution can be found using the following settings:
# All defaults

# The equation is defined on the domain [0,1], with the boundary
# conditions defined at x=0 and x=1.

from math import exp, sin, cos

# Define the original differential equation, assumed to be in the form
# G(x,y,dy/dx) = d2y_dx2 + 1/5*dy_dx + y + 1/5*exp(-x/5)*cos(x) = 0
# Solution is y(x) = y(x) = exp(-x/5)*sin(x)
def Gf(x, y, dy_dx, d2y_dx2):
    return d2y_dx2 + 1/5*dy_dx + y + 1/5*exp(-x/5)*cos(x)

# Boundary conditions
bc0 = 0
bc1 = sin(1)*exp(-1/5) # 0.688938...

# Derivatives of ODE

def dG_dyf(x, y, dy_dx, d2y_dx2):
    return 1

def dG_dydxf(x, y, dy_dx, d2y_dx2):
    return 1/5

def dG_d2ydx2f(x, y, dy_dx, d2y_dx2):
    return 1

# Define the analytical solution.
def yaf(x):
    return exp(-x/5)*sin(x)

# Define the 1st analytical derivative.
def dya_dxf(x):
    return 1/5*exp(-x/5)*(5*cos(x) - sin(x))

# Define the 2nd analytical derivative.
def d2ya_dx2f(x):
    return -2/25*exp(-x/5)*(5*cos(x) + 12*sin(x))
