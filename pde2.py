"""
Base class for 2nd-order partial differential equations

This module provides the base functionality for all 2nd-order partial
differential equation objects used in the nnode software.

Example:
    Create an empty PDE2 object.
        pde1 = PDE2()

Attributes:
    None

Methods:
    None

Todo:
    * Expand base functionality.
"""


from pde import PDE


class PDE2(PDE):
    """Base class for 1st-order PDEs"""
    pass


if __name__ == '__main__':
    pde2 = PDE2()
    print(pde2)
