"""
PDE1IVP - Base class for 1st-order partial differential equation initial-
value problems

This module provides the base functionality for all 1st-order partial
differential equation initial-value problem objects used in the nnode software.

Example:
    Create an empty PDE1IVP object.
        pde1ivp = PDE1IVP()
    Create an PDE1IVP object from a Python module.
        pde1ivp = PDE1IVP(modname)

The solution is assumed to be a function of m independent variables. In the
methods below, x is a vector of independent variables, and delY is the
Jacobian of the solution wrt the independent variables.

Attributes:
    name - String containing name of equation definition module
    G - Function for equation
    bc[] - Vector of boundary condition functions for each variable;
      only first element in each row is used since IVP
    dG_dY - Function for derivative of Gf wrt Y
    dG_ddelY - Array of function for derivative of Gf wrt dY/dx[j]
    Ya - (Optional) function for analytical solution Ya(x)
    delYa - (Optional) Array of functions for analytical Jacobian of solution

Methods:

Todo:
"""


from importlib import import_module

from pde1 import PDE1


class PDE1IVP(PDE1):
    """Base class for all 1st-order partial differential equation initial-
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
        self.bc = None
        self.dG_dY = None
        self.dG_ddelY = None
        self.Ya = None
        self.delYa = None
        if diffeqmod:
            self.name = diffeqmod
            pdemod = import_module(diffeqmod)
            assert pdemod.G
            assert pdemod.bc is not None
            assert pdemod.dG_dY
            assert pdemod.dG_ddelY
            self.G = pdemod.G
            self.bc = pdemod.bc
            self.dG_dY = pdemod.dG_dY
            self.dG_ddelY = pdemod.dG_ddelY
            if pdemod.Ya:
                self.Ya = pdemod.Ya
            if pdemod.delYa:
                self.delYa = pdemod.delYa


if __name__ == '__main__':
    pde1ivp = PDE1IVP()
    print(pde1ivp)
