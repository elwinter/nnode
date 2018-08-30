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
    * Expand base functionality.
"""

from ode import ODE

class ODE1(ODE):
    """Base class for all 1st-order ordinary differential equation objects"""
    pass

if __name__ == '__main__':
    deq = ODE1()
    print(deq)
