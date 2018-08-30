"""
ODE1IVP - Base class for 1st-order ordinary differential equation initial-
value problems

This module provides the base functionality for all 1st-order ordinary
differential equation initial-value problem objects used in the nnode software.

Example:
    Create an empty ODE1IVP object.
        ode1ivp = ODE1IVP()
    Create an ODE1IVP object from a Python module.
        ode1ivp = ODE1IVP(modname)

Attributes:
    name - String containing name of equation definition module
    Gf - Function for equation
    ic - Scalar for initial condition (y(x) = y(0))
    dG_dyf - Function for derivative of Gf wrt y
    dg_dydxf - Function for derivative of Gf wrt dy/dx
    yaf - (Optional) function for analytical solution y(x)
    dya_dxf - (Optional) function for analytical derivative dy/dx

Methods:
    __init__
    __str__

Todo:
    * Expand base functionality.
"""

from importlib import import_module
from inspect import getsource

from ode1 import ODE1

class ODE1IVP(ODE1):
    """Base class for all 1st-order ordinary differential equation initial-
    value problem objects"""

    def __init__(self, diffeqmod=None):
        self.name = None
        self.Gf = None
        self.ic = None
        self.dG_dyf = None
        self.dG_dydxf = None
        self.yaf = None
        self.dya_dxf = None
        if diffeqmod:
            self.name = diffeqmod
            odemod = import_module(diffeqmod)
            assert odemod.Gf          # Function for the ODE as a whole
            assert odemod.ic is not None  # Initial condition at x=0
            assert odemod.dG_dyf      # Function for derivative of G wrt y
            assert odemod.dG_dydxf    # Function for derivative of G wrt dy/dx
            # yaf() is the optional function for analytical solution ya
            # dya_dxf is the optional function for analytical derivative dya/dx
            self.Gf = odemod.Gf
            self.ic = odemod.ic
            self.dG_dyf = odemod.dG_dyf
            self.dG_dydxf = odemod.dG_dydxf
            if odemod.yaf:
                self.yaf = odemod.yaf
            if odemod.dya_dxf:
                self.dya_dxf = odemod.dya_dxf

    def __str__(self):
        s = ''
        s += 'ODE1IVP:\n'
        s += "name = %s\n" % self.name
        s += "Gf = %s\n"  % (getsource(self.Gf).rstrip() if self.Gf else None)
        s += "dG_dyf = %s\n"  % (getsource(self.dG_dyf).rstrip()
                                 if self.dG_dydxf else None)
        s += "dG_dydxf = %s\n"  % (getsource(self.dG_dydxf).rstrip()
                                   if self.dG_dydxf else None)
        s += "ic = %s\n"  % (self.ic if self.ic is not None else None)
        s += "yaf = %s\n"  % (getsource(self.yaf).rstrip() if self.yaf else None)
        s += "dya_dxf = %s\n"  % (getsource(self.dya_dxf).rstrip()
                                  if self.dya_dxf else None)
        return s.rstrip()  # Strip trailing newline if any.

if __name__ == '__main__':
    deq = ODE1IVP()
    print(deq)
