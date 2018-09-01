"""
ODE2IVP - Base class for 2nd-order ordinary differential equation initial
value problems

This module provides the base functionality for all 2nd-order ordinary
differential equation initial value problem objects used in the nnode
software.

Example:
    Create an empty ODE2IVP object.
        ode2ivp = ODE2IVP()
    Create an ODE2IVP object from a Python module.
        ode2ivp = ODE2IVP(modname)

Attributes:
    name - String containing name of equation definition module
    Gf - Function for equation
    ic - Scalar for initial condition at x=0 (y(x) = y(0))
    ic1 - Scalar for initial derivative condition at x=0 (dy/dx(x) = dy/dx(0))
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

class ODE2IVP(ODE2):
    """Base class for all 2nd-order ordinary differential equation initial-
    value problem objects"""

    def __init__(self, diffeqmod=None):
        self.name = None
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
            self.name = diffeqmod
            odemod = import_module(diffeqmod)
            assert odemod.Gf          # Function for the ODE as a whole
            assert odemod.ic is not None   # y(0)
            assert odemod.ic1 is not None  # dy/dx (0)
            assert odemod.dG_dyf      # Function for derivative of G wrt y
            assert odemod.dG_dydxf    # Function for derivative of G wrt dy/dx
            assert odemod.dG_d2ydx2f  # Function for derivative of G wrt d2y/dx2
            # yaf() is the optional function for analytical solution ya
            # dya_dxf is the optional function for analytical derivative dya/dx
            # d2ya_dx2f is the optional function for analytical derivative
            # d2ya/dx2
            self.Gf = odemod.Gf
            self.ic = odemod.ic
            self.ic1 = odemod.ic1
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
        s += "Gf = %s\n"  % (getsource(self.Gf).rstrip() if self.Gf else None)
        s += "dG_dyf = %s\n"  % (getsource(self.dG_dyf).rstrip()
                                 if self.dG_dydxf else None)
        s += "dG_dydxf = %s\n"  % (getsource(self.dG_dydxf).rstrip()
                                   if self.dG_dydxf else None)
        s += "ic0 = %s\n"  % (self.ic if self.ic is not None else None)
        s += "ic1 = %s\n"  % (self.ic1 if self.ic1 is not None else None)
        s += "yaf = %s\n"  % (getsource(self.yaf).rstrip() if self.yaf else None)
        s += "dya_dxf = %s\n"  % (getsource(self.dya_dxf).rstrip()
                                  if self.dya_dxf else None)
        s += "d2ya_dx2f = %s\n"  % (getsource(self.d2ya_dx2f).rstrip()
                                    if self.d2ya_dx2f else None)
        return s.rstrip()  # Strip trailing newline if any.

if __name__ == '__main__':
    ode2ivp = ODE2IVP()
    print(ode2ivp)
