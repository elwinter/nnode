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
    * Expand base functionality.
"""

from diffeq import DiffEq

class ODE(DiffEq):
    """Base class for all ordinary differential equation objects"""
    pass

if __name__ == '__main__':
    deq = ODE()
    print(deq)
