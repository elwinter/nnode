"""
sigma - Python module to implement the sigma transfer function and derivatives

This module provides the sigma transfer function and derivatives.

Example:
    Calculate the sigma for z=1.75.
        s = sigma(1.75)

Attributes:
    None

Methods:
    sigma()
    dsigma_dz()
    d2sigma_dz2()
    d3sigma_dz3()
    d4sigma_dz4()

Todo:
    None
"""


from math import exp


def sigma(z):
    """Sigma transfer function"""
    return 1 / (1 + exp(-z))


def dsigma_dz(z):
    """Sigma transfer function 1st derivative"""
    return exp(-z)/(1 + exp(-z))**2


def d2sigma_dz2(z):
    """Sigma transfer function 2nd derivative"""
    return 2*exp(-2*z) / (1 + exp(-z))**3 - exp(-z)/(1 + exp(-z))**2


def d3sigma_dz3(z):
    """Sigma transfer function 3rd derivative"""
    return \
        6*exp(-3*z)/(1 + exp(-z))**4 - 6 * exp(-2*z)/(1 + exp(-z))**3 \
        + exp(-z)/(1 + exp(-z))**2


def d4sigma_dz4(z):
    """Sigma transfer function 4th derivative"""
    return \
        24*exp(-4*z)/(1 + exp(-z))**5 - 36*exp(-3*z)/(1 + exp(-z))**4 + \
        14*exp(-2*z)/(1 + exp(-z))**3 - exp(-z)/(1 + exp(-z))**2


if __name__ == '__main__':
    z = 1
    print("sigma(%g) = %g" % (z, sigma(z)))
    print("dsigma_dz(%g) = %g" % (z, dsigma_dz(z)))
    print("d2sigma_dz2(%g) = %g" % (z, d2sigma_dz2(z)))
    print("d3sigma_dz3(%g) = %g" % (z, d3sigma_dz3(z)))
    print("d4sigma_dz4(%g) = %g" % (z, d4sigma_dz4(z)))
