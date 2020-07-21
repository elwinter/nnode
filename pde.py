"""
PDE - Base class for partial differential equations

This module provides the base functionality for all partial differential
equation objects used in the nnode software.

Example:
    Create an empty PDE object.
        pde = PDE()

Attributes:
    None

Methods:
    None

Todo:
    None
"""


from diffeq import DiffEq


class PDE(DiffEq):
    """Base class for all partial differential equation objects"""

    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    pde = PDE()
    print(pde)
