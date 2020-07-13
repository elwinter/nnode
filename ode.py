"""
ODE - Base class for ordinary differential equations

This module provides the base functionality for all ordinary differential
equation objects used in the nnode software.

Example:
    Create an empty ODE object.
        ode = ODE()

Attributes:
    None

Methods:
    None

Todo:
    None
"""


from diffeq import DiffEq


class ODE(DiffEq):
    """Base class for all ordinary differential equation objects"""

    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    ode = ODE()
    print(ode)
