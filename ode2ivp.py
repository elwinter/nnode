from importlib import import_module

from ode2 import ODE2

class ODE2IVP(ODE2):

    def __init__(self, diffeqmod = None):
        """Create a ODE1IVP object from a Python module."""
        self.Gf = None
        self.ic = None
        self.ic1 = None
        self.dG_dyf = None
        self.dG_dydxf = None
        self.dG_d2ydx2f = None
        self.yaf = None
        self.dya_dxf = None
        self.d2ya_dx2f = None
        if diffeqmod:
            odemod = import_module(diffeqmod)
            assert odemod.Gf          # Function for the ODE as a whole
            assert odemod.ic != None  # y(0)
            assert odemod.ic1 != None  # dy/dx (0)
            assert odemod.dG_dyf      # Function for derivative of G wrt y
            assert odemod.dG_dydxf    # Function for derivative of G wrt dy/dx
            assert odemod.dG_d2ydx2f  # Function for derivative of G wrt d2y/dx2
            assert odemod.yaf         # Function for analytical solution ya
            assert odemod.dya_dxf     # Function for analytical derivative dya/dx
            assert odemod.d2ya_dx2f   # Function for analytical derivative dya/dx
            self.Gf = odemod.Gf
            self.ic = odemod.ic
            self.ic1 = odemod.ic1
            self.dG_dyf = odemod.dG_dyf
            self.dG_dydxf = odemod.dG_dydxf
            self.dG_d2ydx2f = odemod.dG_d2ydx2f
            self.yaf = odemod.yaf
            self.dya_dxf = odemod.dya_dxf
            self.d2ya_dx2f = odemod.d2ya_dx2f

if __name__ == '__main__':
    ode2ivp = ODE2IVP()
    print(ode2ivp)
