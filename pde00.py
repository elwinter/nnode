# Sample 1st-order PDE IVP

# A reasonable solution can be found using the following settings:
# All defaults

# The equation is defined on the domain [[0,1],[0,1]], with the
# initial conditions specified at x=0 and y=0.

# Define the original differential equation, assumed to be in the form
# G(x,y,psi,dpsi/dx,dpsi/dy) = x*y - psi = 0.
# The analytical solution is: psi(x,y) = x*y
def Gf(xy, psi, del_psi):
    (x, y) = xy
    (dpsi_dx, dpsi_dy) = del_psi
    return x * y - psi

# First partials of the PDE

def dG_dxf(xy, psi, del_psi):
    (x, y) = xy
    (dpsi_dx, dpsi_dy) = del_psi
    return y

def dG_dyf(xy, psi, del_psi):
    (x, y) = xy
    (dpsi_dx, dpsi_dy) = del_psi
    return x

del_Gf = ( dG_dxf, dG_dyf )

def dG_dpsif(xy, psi, del_psi):
    return -1

def dG_dpsi_dxf(xy, psi, del_psi):
    return 0

def dG_dpsi_dyf(xy, psi, del_psi):
    return 0

dG_ddel_psif = ( dG_dpsi_dxf, dG_dpsi_dyf )

# Boundary condition functions and derivatives
def f0f(y):
    return 0

def g0f(x):
    return 0

bcf = ( f0f, g0f )

def df0_dyf(y):
    return 0

def dg0_dxf(x):
    return 0

bcdf = ( df0_dyf, dg0_dxf)

# Define the analytical solution and its derivatives.
def psiaf(xy):
    (x, y) = xy
    return x * y

def dpsia_dxf(xy):
    (x, y) = xy
    return y

def dpsia_dyf(xy):
    (x, y) = xy
    return x

del_psiaf = ( dpsia_dxf, dpsia_dyf )
