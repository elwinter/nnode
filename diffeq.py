"""
DiffEq - Base class for differential equations

This module provides the base functionality for all differential equation
objects used in the nnode software.

Example:
    Create an empty DiffEq object.
        diffeq = DiffEq()

Attributes:
    G - Function for differential equation, in the form G() = 0

Methods:
    None

Todo:
    None
"""


class DiffEq:
    """Base class for all differential equation objects"""

    def __init__(self):
        pass

    def G(self):
        return None


if __name__ == '__main__':
    diffeq = DiffEq()
    print(diffeq)
