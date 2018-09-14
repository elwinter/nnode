"""
ODE2 - Base class for 2nd-order ordinary differential equations

This module provides the base functionality for all 2nd-order ordinary
differential equation objects used in the nnode software.

Example:
    Create an empty ODE2 object.
        ode2 = ODE2()

Attributes:
    None

Methods:
    None

Todo:
    * Expand base functionality.
"""


from ode import ODE


class ODE2(ODE):
    """Base class for all 2nd-order ordinary differential equation objects"""
    pass


if __name__ == '__main__':
    ode2 = ODE2()
    print(ode2)
