from importlib import import_module

from ode1 import ODE1

class ODE1IVP(ODE1):

    def __init__(self, diffeqmod = None):
        """Create a ODE1IVP object from a Python module."""
        self.Gf = None
        self.ic = None
        self.dG_dyf = None
        self.dG_dydxf = None
        self.yaf = None
        self.dya_dxf = None
        if diffeqmod:
            odemod = import_module(diffeqmod)
            assert odemod.Gf          # Function for the ODE as a whole
            assert odemod.ic != None  # Initial condition at x=0
            assert odemod.dG_dyf      # Function for derivative of G wrt y
            assert odemod.dG_dydxf    # Function for derivative of G wrt dy/dx
            assert odemod.yaf         # Function for analytical solution ya
            assert odemod.dya_dxf     # Function for analytical derivative dya/dx
            self.Gf = odemod.Gf
            self.ic = odemod.ic
            self.dG_dyf = odemod.dG_dyf
            self.dG_dydxf = odemod.dG_dydxf
            self.yaf = odemod.yaf
            self.dya_dxf = odemod.dya_dxf
