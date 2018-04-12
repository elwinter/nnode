from math import exp

# Define the sigmoid transfer function and its derivatives.
def sigma(z):
    return 1 / (1 + exp(-z))

def dsigma_dz(z):
    return exp(-z)/(1 + exp(-z))**2

def d2sigma_dz2(z):
    return 2*exp(-2*z) / (1 + exp(-z))**3 - exp(-z)/(1 + exp(-z))**2

def d3sigma_dz3(z):
    return (
        6*exp(-3*z)/(1 + exp(-z))**4 - 6 * exp(-2*z)/(1 + exp(-z))**3
        + exp(-z)/(1 + exp(-z))**2
    )

def d4sigma_dz4(z):
    return (
        24*exp(-4*z)/(1 + exp(-z))**5 - 36*exp(-3*z)/(1 + exp(-z))**4 +
        14*exp(-2*z)/(1 + exp(-z))**3 - exp(-z)/(1 + exp(-z))**2
    )
