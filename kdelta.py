"""
kdelta - Python module to implement the Kronecker delta function

This module provides the Kronecker delta function.

Example:
    Calculate the Kronecker delta for 2 integers.
        kd = kdelta(i, j)

Attributes:
    None

Methods:
    None

Todo:
    None
"""


def kdelta(i, j):
    """Return 1 if i == j, else 0."""
    return 1 if i == j else 0
