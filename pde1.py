"""
PDE1 - Base class for 1st-order partial differential equations

This module provides the base functionality for all 1st-order partial
differential equation objects used in the nnode software.

Example:
    Create an empty ODE1 object.
        pde1 = PDE1()

The solution is assumed to be a function of m independent variables. In the
methods below, x is a vector of independent variables, and delY is the
Jacobian of the solution wrt the independent variables.

Attributes:
    None

Methods:
    None

Todo:
    None
"""


from pde import PDE


class PDE1(PDE):
    """Base class for all 1st-order partial differential equation objects"""

    def __init__(self):
        super().__init__()

    def G(self, x, Y, delY):
        return None



if __name__ == '__main__':
    pde1 = PDE1()
    print(pde1)
