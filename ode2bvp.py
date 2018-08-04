from importlib import import_module

from ode2 import ODE2

class ODE2BVP(ODE2):

    def __init__(self, diffeqmod = None):
        """Create a ODE2BVP object from a Python module."""
        self.Gf = None
        self.bc0 = None
        self.bc1 = None
        self.dG_dyf = None
        self.dG_dydxf = None
        self.dG_d2ydx2f = None
        self.yaf = None
        self.dya_dxf = None
        self.d2ya_dx2f = None
        if diffeqmod:
            odemod = import_module(diffeqmod)
            assert odemod.Gf          # Function for the ODE as a whole
            assert odemod.bc0 != None  # y(0)
            assert odemod.bc1 != None  # y(1)
            assert odemod.dG_dyf      # Function for derivative of G wrt y
            assert odemod.dG_dydxf    # Function for derivative of G wrt dy/dx
            assert odemod.dG_d2ydx2f  # Function for derivative of G wrt d2y/dx2
            assert odemod.yaf         # Function for analytical solution ya
            assert odemod.dya_dxf     # Function for analytical derivative dya/dx
            assert odemod.d2ya_dx2f   # Function for analytical derivative dya/dx
            self.Gf = odemod.Gf
            self.bc0 = odemod.bc0
            self.bc1 = odemod.bc1
            self.dG_dyf = odemod.dG_dyf
            self.dG_dydxf = odemod.dG_dydxf
            self.dG_d2ydx2f = odemod.dG_d2ydx2f
            self.yaf = odemod.yaf
            self.dya_dxf = odemod.dya_dxf
            self.d2ya_dx2f = odemod.d2ya_dx2f

if __name__ == '__main__':
    ode2bvp = ODE2BVP()
    print(ode2bvp)
