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
    s()
    s1()
    s2()
    s3()
    s4()

Todo:
    None
"""


from math import exp


def sigma(z):
    """Sigma transfer function"""
    return 1/(1 + exp(-z))


def dsigma_dz(z):
    """Sigma transfer function 1st derivative"""
    return exp(-z)/(1 + exp(-z))**2


def d2sigma_dz2(z):
    """Sigma transfer function 2nd derivative"""
    return 2*exp(-2*z)/(1 + exp(-z))**3 - exp(-z)/(1 + exp(-z))**2


def d3sigma_dz3(z):
    """Sigma transfer function 3rd derivative"""
    return (6*exp(-3*z)/(1 + exp(-z))**4 - 6 * exp(-2*z)/(1 + exp(-z))**3
            + exp(-z)/(1 + exp(-z))**2)


def d4sigma_dz4(z):
    """Sigma transfer function 4th derivative"""
    return (24*exp(-4*z)/(1 + exp(-z))**5 - 36*exp(-3*z)/(1 + exp(-z))**4
            + 14*exp(-2*z)/(1 + exp(-z))**3 - exp(-z)/(1 + exp(-z))**2)


# Alternative forms as a function of sigma itself

def s(z):
    """Sigma transfer function"""
    return 1/(1 + exp(-z))


def s1(s):
    """Sigma transfer function 1st derivative"""
    return s - s**2


def s2(s):
    """Sigma transfer function 2nd derivative"""
    return 2*s**3 - 3*s**2 + s


def s3(s):
    """Sigma transfer function 3rd derivative"""
    return -6*s**4 + 12*s**3 - 7*s**2 + s


def s4(s):
    """Sigma transfer function 4th derivative"""
    return 24*s**5 - 60*s**4 + 50*s**3 - 15*s**2 + s


if __name__ == '__main__':
    z = 1
    
    print("sigma(%g) = %g" % (z, sigma(z)))
    print("dsigma_dz(%g) = %g" % (z, dsigma_dz(z)))
    print("d2sigma_dz2(%g) = %g" % (z, d2sigma_dz2(z)))
    print("d3sigma_dz3(%g) = %g" % (z, d3sigma_dz3(z)))
    print("d4sigma_dz4(%g) = %g" % (z, d4sigma_dz4(z)))

    print("s(%g)  = %g" % (z, s(z)))
    print("s1(%g) = %g" % (z, s1(s(z))))
    print("s2(%g) = %g" % (z, s2(s(z))))
    print("s3(%g) = %g" % (z, s3(s(z))))
    print("s4(%g) = %g" % (z, s4(s(z))))
