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


if __name__ == '__main__':
    for i in range(3):
        for j in range(3):
            kd = kdelta(i, j)
            print(i, j, kd)
            if i == j:
                assert kd == 1
            else:
                assert kd == 0
