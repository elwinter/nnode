"""
Base class for 1st-order partial differential equations

This module provides the base functionality for all 1st-order partial
differential equation objects used in the nnode software.

Example:
    Create an empty PDE1 object.
        pde1 = PDE1()

Attributes:
    None

Methods:
    None

Todo:
    * Expand base functionality.
"""


from pde import PDE


class PDE1(PDE):
    """Base class for 1st-order PDEs"""
    pass


if __name__ == '__main__':
    pde1 = PDE1()
    print(pde1)
