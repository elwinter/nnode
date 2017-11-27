# Sample 1st-order PDE IVP

# Define the analytical solution and its derivatives.
def psia(xv):
    (x, y) = xv
    return x * y

def dpsia_dx0(xv):
    (x, y) = xv
    return y

def dpsia_dx1(xv):
    (x, y) = xv
    return x

dpsia_dx = [ dpsia_dx0, dpsia_dx1 ]

def d2psia_dx02(xv):
    (x, y) = xv
    return 0

def d2psia_dx12(xv):
    (x, y) = xv
    return 0

d2psia_dx2 = [ d2psia_dx02, d2psia_dx12 ]

# Define the original differential equation and itss derivatives.
def fG(xv, psi, dpsi_dxv):
    (x, y) = xv
    (dpsi_dx, dpsi_dy) = dpsi_dxv
    return dpsi_dx * dpsi_dy - x * y

def dfG_dx0(xv, psi, dpsi_dxv):
    (x, y) = xv
    (dpsi_dx, dpsi_dy) = dpsi_dxv
    return -y

def dfG_dx1(xv, psi, dpsi_dxv):
    (x, y) = xv
    (dpsi_dx, dpsi_dy) = dpsi_dxv
    return -x

dfG_dx = [ dfG_dx0, dfG_dx1 ]

def d2fG_dx02(xv, psi, dpsi_dxv):
    (x, y) = xv
    (dpsi_dx, dpsi_dy) = dpsi_dxv
    return 0

def d2fG_dx12(xv, psi, dpsi_dxv):
    (x, y) = xv
    (dpsi_dx, dpsi_dy) = dpsi_dxv
    return 0

d2fG_dx2 = [ d2fG_dx02, d2fG_dx12 ]

def dfG_dpsi(xv, psi, dpsi_dxv):
    (x, y) = xv
    (dpsi_dx, dpsi_dy) = dpsi_dxv
    return 0

def d2fG_dpsi2(xv, psi, dpsi_dxv):
    (x, y) = xv
    (dpsi_dx, dpsi_dy) = dpsi_dxv
    return 0

def dfG_dpsi_dx0(xv, psi, dpsi_dxv):
    (x, y) = xv
    (dpsi_dx, dpsi_dy) = dpsi_dxv
    return dpsi_dy

def dfG_dpsi_dx1(xv, psi, dpsi_dxv):
    (x, y) = xv
    (dpsi_dx, dpsi_dy) = dpsi_dxv
    return dpsi_dx

dfG_dpsi_dx = [ dfG_dpsi_dx0, dfG_dpsi_dx1 ]

def d2fG_dpsi_dx02(xv, psi, dpsi_dxv):
    (x, y) = xv
    (dpsi_dx, dpsi_dy) = dpsi_dxv
    return 0

def d2fG_dpsi_dx12(xv, psi, dpsi_dxv):
    (x, y) = xv
    (dpsi_dx, dpsi_dy) = dpsi_dxv
    return 0

d2fG_dpsi_dx2 = [ d2fG_dpsi_dx02, d2fG_dpsi_dx12 ]

# Boundary conditions
xmin = 0
xmax = 1

def f0(y):
    return 0

def g0(x):
    return 0

bc = [ f0, g0 ]

def df0_dy(y):
    return 0

def dg0_dx(x):
    return 0

bcd = [ df0_dy, dg0_dx]

def d2f0_dy2(y):
    return 0

def d2g0_dx2(x):
    return 0

bcdd = [ d2f0_dy2, d2g0_dx2]
