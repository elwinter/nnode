"""
ODE2BVP - Base class for 2nd-order ordinary differential equation boundary
value problems

This module provides the base functionality for all 2nd-order ordinary
differential equation boundary value problem objects used in the nnode
software.

Example:
    Create an empty ODE2BVP object.
        ode2bvp = ODE2BVP()
    Create an ODE2BVP object from a Python module.
        ode2bvp = ODE2BVP(modname)

Attributes:
    name - String containing name of equation definition module
    Gf - Function for equation
    bc0 - Scalar for boundary condition at x=0
    bc1 - Scalar for boundary condition at x=1
    dG_dYf - Function for derivative of Gf wrt Y
    dg_ddYdxf - Function for derivative of Gf wrt dY/dx
    dg_dd2Ydx2f - Function for derivative of Gf wrt d2Y/dx2
    Yaf - (Optional) function for analytical solution Ya(x)
    dYa_dxf - (Optional) function for analytical derivative dY/dx
    d2Ya_dx2f - (Optional) function for analytical derivative d2Y/dx2

Methods:

Todo:
"""


from importlib import import_module

from ode2 import ODE2


class ODE2BVP(ODE2):
    """Base class for all 2nd-order ordinary differential equation boundary-
    value problem objects"""

    def __init__(self, diffeqmod=None):
        """
        Constructor
        Parameters:
        diffeqmod - The name of the Python module containing the problem definition.
        """
        self.name = None
        self.Gf = None
        self.bc0 = None
        self.bc1 = None
        self.dG_dYf = None
        self.dG_ddydxf = None
        self.dG_dd2Ydx2f = None
        self.Yaf = None
        self.dYa_dxf = None
        self.d2Ya_dx2f = None
        if diffeqmod:
            self.name = diffeqmod
            odemod = import_module(diffeqmod)
            assert odemod.Gf          # Function for the ODE as a whole
            assert odemod.bc0 is not None  # y(0)
            assert odemod.bc1 is not None  # y(1)
            assert odemod.dG_dYf      # Function for deriv of G wrt Y
            assert odemod.dG_ddYdxf    # Function for deriv of G wrt dY/dx
            assert odemod.dG_dd2Ydx2f  # Function for deriv of G wrt d2Y/dx2
            self.Gf = odemod.Gf
            self.bc0 = odemod.bc0
            self.bc1 = odemod.bc1
            self.dG_dYf = odemod.dG_dYf
            self.dG_ddYdxf = odemod.dG_ddYdxf
            self.dG_dd2Ydx2f = odemod.dG_dd2Ydx2f
            if odemod.Yaf:
                self.Yaf = odemod.Yaf
            if odemod.dYa_dxf:
                self.dYa_dxf = odemod.dYa_dxf
            if odemod.d2Ya_dx2f:
                self.d2Ya_dx2f = odemod.d2Ya_dx2f


if __name__ == '__main__':
    ode2bvp = ODE2BVP()
    print(ode2bvp)
    ode2bvp = ODE2BVP('eq.lagaris_03_bvp')
    print(ode2bvp)
