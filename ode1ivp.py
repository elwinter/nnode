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
    G - Function for equation
    ic - Scalar for initial condition Y(0)
    dG_dY - Function for derivative of Gf wrt Y
    dG_ddYdx - Function for derivative of Gf wrt dY/dx
    Ya - (Optional) function for analytical solution Ya(x)
    dYa_dx - (Optional) function for analytical derivative dY_x(x)

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
        super().__init__()
        self.name = None
        self.G = None
        self.ic = None
        self.dG_dY = None
        self.dG_ddYdx = None
        self.Ya = None
        self.dYa_dx = None
        if diffeqmod:
            self.name = diffeqmod
            odemod = import_module(diffeqmod)
            assert odemod.G
            assert odemod.ic is not None
            assert odemod.dG_dY
            assert odemod.dG_ddYdx
            self.G = odemod.G
            self.ic = odemod.ic
            self.dG_dY = odemod.dG_dY
            self.dG_ddYdx = odemod.dG_ddYdx
            if odemod.Ya:
                self.Ya = odemod.Ya
            if odemod.dYa_dx:
                self.dYa_dx = odemod.dYa_dx


if __name__ == '__main__':
    ode1ivp = ODE1IVP()
    print(ode1ivp)
    ode1ivp = ODE1IVP('lagaris_01')
    print(ode1ivp)
