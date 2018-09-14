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
    bc0 - Scalar for boundary condition at x=0 (y(x) = y(0))
    bc1 - Scalar for boundary condition at x=1 (y(x) = y(1))
    dG_dyf - Function for derivative of Gf wrt y
    dg_dydxf - Function for derivative of Gf wrt dy/dx
    dg_d2ydx2f - Function for derivative of Gf wrt d2y/dx2
    yaf - (Optional) function for analytical solution y(x)
    dya_dxf - (Optional) function for analytical derivative dy/dx
    d2ya_dx2f - (Optional) function for analytical derivative d2y/dx2

Methods:
    __init__
    __str__

Todo:
    * Expand base functionality.
"""


from importlib import import_module
from inspect import getsource

from ode2 import ODE2


class ODE2BVP(ODE2):
    """Base class for all 2nd-order ordinary differential equation boundary-
    value problem objects"""

    def __init__(self, diffeqmod=None):
        self.name = None
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
            self.name = diffeqmod
            odemod = import_module(diffeqmod)
            assert odemod.Gf          # Function for the ODE as a whole
            assert odemod.bc0 is not None  # y(0)
            assert odemod.bc1 is not None  # y(1)
            assert odemod.dG_dyf      # Function for deriv of G wrt y
            assert odemod.dG_dydxf    # Function for deriv of G wrt dy/dx
            assert odemod.dG_d2ydx2f  # Function for deriv of G wrt d2y/dx2
            # yaf() is the optional function for analytical solution ya
            # dya_dxf is the optional function for analytical derivative dya/dx
            # d2ya_dx2f is the optional function for analytical derivative
            # d2ya/dx2
            self.Gf = odemod.Gf
            self.bc0 = odemod.bc0
            self.bc1 = odemod.bc1
            self.dG_dyf = odemod.dG_dyf
            self.dG_dydxf = odemod.dG_dydxf
            self.dG_d2ydx2f = odemod.dG_d2ydx2f
            if odemod.yaf:
                self.yaf = odemod.yaf
            if odemod.dya_dxf:
                self.dya_dxf = odemod.dya_dxf
            if odemod.d2ya_dx2f:
                self.d2ya_dx2f = odemod.d2ya_dx2f

    def __str__(self):
        s = ''
        s += 'ODE2BVP:\n'
        s += "name = %s\n" % self.name
        s += "Gf = %s\n" % (getsource(self.Gf).rstrip() if self.Gf else None)
        s += "dG_dyf = %s\n" % (getsource(self.dG_dyf).rstrip()
                                if self.dG_dyf else None)
        s += "dG_dydxf = %s\n" % (getsource(self.dG_dydxf).rstrip()
                                  if self.dG_dydxf else None)
        s += "bc0 = %s\n" % (self.bc0 if self.bc0 is not None else None)
        s += "bc1 = %s\n" % (self.bc1 if self.bc1 is not None else None)
        s += "yaf = %s\n" % (getsource(self.yaf).rstrip()
                             if self.yaf else None)
        s += "dya_dxf = %s\n" % (getsource(self.dya_dxf).rstrip()
                                 if self.dya_dxf else None)
        s += "d2ya_dx2f = %s\n" % (getsource(self.d2ya_dx2f).rstrip()
                                   if self.d2ya_dx2f else None)
        return s.rstrip()  # Strip trailing newline if any.


if __name__ == '__main__':
    ode2bvp = ODE2BVP()
    print(ode2bvp)
