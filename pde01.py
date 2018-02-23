# Sample 1st-order PDE IVP

# Define the analytical solution and its derivatives.
def psia(xv):
    (x, y) = xv
    return x**2 + y**2

def dpsia_dx0(xv):
    (x, y) = xv
    return 2 * x

def dpsia_dx1(xv):
    (x, y) = xv
    return 2 * y

dpsia_dx = [ dpsia_dx0, dpsia_dx1 ]

def d2psia_dx02(xv):
    (x, y) = xv
    return 2

def d2psia_dx12(xv):
    (x, y) = xv
    return 2

d2psia_dx2 = [ d2psia_dx02, d2psia_dx12 ]

# Define the original differential equation and itss derivatives.
def fG(xv, psi, dpsi_dxv):
    (x, y) = xv
    (dpsi_dx, dpsi_dy) = dpsi_dxv
    return 0.5 * (x * dpsi_dx + y * dpsi_dy) - psi

def dfG_dx0(xv, psi, dpsi_dxv):
    (x, y) = xv
    (dpsi_dx, dpsi_dy) = dpsi_dxv
    return 0.5 * dpsi_dx

def dfG_dx1(xv, psi, dpsi_dxv):
    (x, y) = xv
    (dpsi_dx, dpsi_dy) = dpsi_dxv
    return 0.5 * dpsi_dy

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
    return -1

def d2fG_dpsi2(xv, psi, dpsi_dxv):
    (x, y) = xv
    (dpsi_dx, dpsi_dy) = dpsi_dxv
    return 0

def dfG_dpsi_dx0(xv, psi, dpsi_dxv):
    (x, y) = xv
    (dpsi_dx, dpsi_dy) = dpsi_dxv
    return 0.5 * x

def dfG_dpsi_dx1(xv, psi, dpsi_dxv):
    (x, y) = xv
    (dpsi_dx, dpsi_dy) = dpsi_dxv
    return 0.5 * y

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
    return y**2

def g0(x):
    return x**2

bc = [ f0, g0 ]

def df0_dy(y):
    return 2 * y

def dg0_dx(x):
    return 2 * x

bcd = [ df0_dy, dg0_dx]

def d2f0_dy2(y):
    return 2

def d2g0_dx2(x):
    return 2

bcdd = [ d2f0_dy2, d2g0_dx2]
