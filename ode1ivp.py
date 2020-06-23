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
    ic - Scalar for initial condition Y(0)
    dG_dYf - Function for derivative of Gf wrt Y
    dG_dYdxf - Function for derivative of Gf wrt dY/dx
    Yaf - (Optional) function for analytical solution Ya(x)
    dYa_dxf - (Optional) function for analytical derivative dY_x(x)

Methods:

Todo:
"""


from importlib import import_module

from ode1 import ODE1


class ODE1IVP(ODE1):
    """Base class for all 1st-order ordinary differential equation initial-
    value problem objects"""

    def __init__(self, diffeqmod=None):
        """
        Constructor
        Parameters:
        diffeqmod - The name of the Python module containing the problem definition.
        """
        self.name = None
        self.Gf = None
        self.ic = None
        self.dG_dYf = None
        self.dG_dYdxf = None
        self.Yaf = None
        self.dYa_dxf = None
        if diffeqmod:
            self.name = diffeqmod
            odemod = import_module(diffeqmod)
            assert odemod.Gf              # Function for the ODE as a whole
            assert odemod.ic is not None  # Initial condition at x=0
            assert odemod.dG_dYf          # Function for derivative of G wrt y
            assert odemod.dG_dYdxf        # Function for derivative of G wrt dy/dx
            self.Gf = odemod.Gf
            self.ic = odemod.ic
            self.dG_dYf = odemod.dG_dYf
            self.dG_dYdxf = odemod.dG_dYdxf
            # Yaf() is the optional function for analytical solution ya
            # dYa_dxf is the optional function for analytical derivative dya/dx
            if odemod.Yaf:
                self.Yaf = odemod.Yaf
            if odemod.dYa_dxf:
                self.dYa_dxf = odemod.dYa_dxf


if __name__ == '__main__':
    ode1ivp = ODE1IVP()
    print(ode1ivp)
    ode1ivp = ODE1IVP('eq.lagaris_01')
    print(ode1ivp)
