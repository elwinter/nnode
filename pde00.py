# Sample 1st-order PDE IVP

# A reasonable solution can be found using ...

# Define the analytical solution.
def ya(x):
    return x[0] * x[1]

# Define the 1st analytical derivatives.
def dya_dx0(x):
    return x[1]

def dya_dx1(x):
    return x[0]

dya_dx = [ dya_dx0, dya_dx1 ]

# Define the 2nd analytical derivatives.
def d2ya_dx02(x):
    return 0

def d2ya_dx12(x):
    return 0

d2ya_dx2 = [ d2ya_dx02, d2ya_dx12 ]

# Define the original differential equation.
def G(x, dy_dx):
    return dy_dx[0] * dy_dx[1] - x[0] * x[1]

# Define the 1st partial derivatives of the differential equation.
def dG_dx0(x, dy_dx):
    return -x[1]

def dG_dx1(x, dy_dx):
    return -x[0]

def dG_dy_dx0(x, dy_dx):
    return dy_dx[1]

def dG_dy_dx1(x, dy_dx):
    return dy_dx[0]

dG_dx = [ dG_dx0, dG_dx1 ]
dG_dy_dx = [ dG_dy_dx0, dG_dy_dx1 ]

# Define the 2nd partial derivatives of the differential equation.
def d2G_dx02(x, y, dy_dx):
    return 0

def d2G_dx12(x, y, dy_dx):
    return 0

def d2G_dy_dx02(x, y, dy_dx):
    return 0

def d2G_dy_dx12(x, y, dy_dx):
    return 0

d2G_dx2 = [ d2G_dx02, d2G_dx12 ]
d2G_dy_dx2 = [ d2G_dy_dx02, d2G_dy_dx12 ]

# Boundary conditions
xmin = 0
xmax = 1

def f0(y):
    return 1

def g0(x):
    return 1

def df0_dy(y):
    return 0

def dg0_dx(x):
    return 0

bc = [ f0, g0 ]
bcd = [ df0_dy, dg0_dx]
