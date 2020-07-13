"""
ODE1 - Base class for 1st-order ordinary differential equations

This module provides the base functionality for all 1st-order ordinary
differential equation objects used in the nnode software.

Example:
    Create an empty ODE1 object.
        ode1 = ODE1()

Attributes:
    None

Methods:
    None

Todo:
    None
"""


from ode import ODE


class ODE1(ODE):
    """Base class for all 1st-order ordinary differential equation objects"""

    def __init__(self):
        super().__init__()

    def G(self, x, Y, dY_dx):
        return None



if __name__ == '__main__':
    ode1 = ODE1()
    print(ode1)
