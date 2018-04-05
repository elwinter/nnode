# Sample 1st-order PDE IVP

# A reasonable solution can be found using the following settings:
# 

# The equation is defined on the domain [[0,1],[0,1]], with the
# initial conditions specified at x=0 and y=0.

# Define the original differential equation, assumed to be in the form
# G(x,y,psi,dpsi/dx,dpsi/dy) = 1/2*(x*dpsi_dx + y*dpsi_dy) - psi = 0.
# The analytical solution is: psi(x,y) = x**2 + y**2
def Gf(xy, psi, del_psi):
    (x, y) = xy
    (dpsi_dx, dpsi_dy) = del_psi
    return 1/2*(x*dpsi_dx + y*dpsi_dy) - psi

# First partials of the PDE

def dG_dxf(xy, psi, del_psi):
    (x, y) = xy
    (dpsi_dx, dpsi_dy) = del_psi
    return 1/2*dpsi_dx

def dG_dyf(xy, psi, del_psi):
    (x, y) = xy
    (dpsi_dx, dpsi_dy) = del_psi
    return 1/2*dpsi_dy

del_Gf = (dG_dxf, dG_dyf)

def dG_dpsif(xy, psi, del_psi):
    (x, y) = xy
    (dpsi_dx, dpsi_dy) = del_psi
    return -1

def dG_dpsi_dxf(xy, psi, del_psi):
    (x, y) = xy
    (dpsi_dx, dpsi_dy) = del_psi
    return x/2

def dG_dpsi_dyf(xy, psi, del_psi):
    (x, y) = xy
    (dpsi_dx, dpsi_dy) = del_psi
    return y/2

dG_ddel_psif = (dG_dpsi_dxf, dG_dpsi_dyf)

# Boundary condition functions and derivatives
def f0f(y):
    return y**2

def g0f(x):
    return x**2

bcf = (f0f, g0f)

def df0_dyf(y):
    return 2*y

def dg0_dxf(x):
    return 2*x

bcdf = (df0_dyf, dg0_dxf)

# Define the analytical solution and its derivatives.
def psiaf(xy):
    (x, y) = xy
    return x**2 + y**2

def dpsia_dxf(xy):
    (x, y) = xy
    return 2*x

def dpsia_dyf(xy):
    (x, y) = xy
    return 2*y

del_psiaf = (dpsia_dxf, dpsia_dyf)
