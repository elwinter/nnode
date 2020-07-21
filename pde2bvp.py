"""
PDE2BVP - Base class for 2nd-order partial differential equation boundary-
value problems

This module provides the base functionality for all 2nd-order partial
differential equation boundary value problem objects used in the nnode software.

Notes:

Y(xv) is the solution to the differential equation (Y is used in place
of the Greek letter psi).

xv is a vector of independent variables,

m is the number of independent variables.

Example:
    Create an empty PDE2BVP object.
        pde2bvp = PDE2BVP()
    Create an PDE2BVP object from a Python module.
        pde2bvp = PDEBIVP(modname)

Attributes:
    name - String containing name of equation definition module
    Gf - Function for equation
    dG_dYf - Function for derivative of Gf wrt Y
    dG_ddelYf - mx1 array of functions for derivatives of Gf wrt dY/dx[j]
    dG_ddeldelYf - mxm array of functions for derivatives of Gf wrt d2Y/dx[jj]dx[j]
    bcf - mx2 rray of functions for boundary conditions on Y(xv)
    delbcf - mx2xm array of functions for boundary condition gradients on Y(xv)
    deldelbcf - mx2xmxm array of functions for boundary condition
    gradient gradients on Y(xv)
    Yaf - (Optional) function for analytical solution Ya(x)
    delYaf - (Optional) mx1 rray of functions for analytical gradients of Ya[xv]
    deldelYaf - (Optional) mxm array of functions for analytical
    gradient gradients of Ya[xv]

Methods:

Todo:
"""

from importlib import import_module

from pde2 import PDE2

class PDE2BVP(PDE2):
    """Base class for all 2nd-order partial differential equation boundary-
    value problem objects"""

    def __init__(self, diffeqmod=None):
        """
        Constructor
        Parameters:
        diffeqmod - The name of the Python module containing the problem definition.
        """
        self.Gf = None
        self.dG_dYf = None
        self.dG_ddelYf = None
        self.dG_ddeldelYf = None
        self.bcf = None
        self.delbcf = None
        self.deldelbcf = None
        self.Yaf = None
        self.delYaf = None
        self.deldelYaf = None
        if diffeqmod:
            pdemod = import_module(diffeqmod)
            assert pdemod.Gf          # Function for the PDE as a whole
            assert pdemod.dG_dYf      # dG/dY   Y=Y(x,y)
            assert pdemod.dG_ddelYf   # dG/dgrad(Y)
            assert pdemod.dG_ddeldelYf  # dG/d(grad(Y)**2)
            assert pdemod.bcf         # Array of initial condition functions
            assert pdemod.delbcf        # Array of initial condition derivatives
            assert pdemod.deldelbcf      # Array of initial condition 2nd derivatives
            self.Gf = pdemod.Gf
            self.dG_dYf = pdemod.dG_dYf
            self.dG_ddelYf = pdemod.dG_ddelYf
            self.dG_ddeldelYf = pdemod.dG_ddeldelYf
            self.bcf = pdemod.bcf
            self.delbcf = pdemod.delbcf
            self.deldelbcf = pdemod.deldelbcf
            if pdemod.Yaf:
                self.Yaf = pdemod.Yaf
            if pdemod.delYaf:
                self.delYaf = pdemod.delYaf
            if pdemod.deldelYaf:
                self.deldelYaf = pdemod.deldelYaf

if __name__ == '__main__':
    pde2bvp = PDE2BVP('eq.lagaris_05')
    print(pde2bvp)
