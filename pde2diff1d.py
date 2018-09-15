"""
PDE2DIFF1D - Base class for 1-D diffusion equations

This module provides the base functionality for all 1-D diffusion equation
objects used in the nnode software.

Example:
    Create an empty PDE2DIFF1D object.
        pde2diff1d = PDE2DIFF1D()
    Create an PDE2DIFF1D object from a Python module.
        pde2diff1d = PDE2DIFF1D(modname)

Attributes:
    name - String containing name of equation definition module
    Gf - Function for equation
    dG_dYf - Function for derivative of Gf wrt Y
    dg_ddelYf - Functions for derivatives of Gf wrt delY
    dg_ddeldelYf - Functions for derivatives of Gf wrt deldelY
    bcf - Functions for boundary conditions
    bcdf - Functions for boundary condition first derivatives
    bcd2f - Functions for boundary condition second derivatives
    Yaf - (Optional) function for analytical solution Y(x,t)
    delYaf - (Optional) function for analytical gradient
    deldelYaff - (Optional) function for analytical Hessian

Methods:
    __init__
    __str__

Todo:
    * Expand base functionality.
"""


from importlib import import_module
from inspect import getsource

from pde2 import PDE2


class PDE2DIFF1D(PDE2):

    def __init__(self, diffeqmod=None):
        self.name = None
        self.Gf = None
        self.dG_dYf = None
        self.dG_ddelYf = None
        self.dG_ddeldelYf = None
        self.bcf = None
        self.bcdf = None
        self.bcd2f = None
        self.Yaf = None
        self.delYaf = None
        self.deldelYaf = None
        if diffeqmod:
            self.name = diffeqmod
            pdemod = import_module(diffeqmod)
            assert pdemod.Gf          # Function for the PDE as a whole
            assert pdemod.dG_dYf      # dG/dY   Y=Y(x,y)
            assert pdemod.dG_ddelYf   # dG/dgrad(Y)
            assert pdemod.dG_ddeldelYf  # dG/d(grad(Y)grad(Y))
            assert pdemod.bcf         # Array of initial condition functions
            assert pdemod.bcdf        # Array of initial condition derivatives
            assert pdemod.bcd2f       # Array of initial condition 2nd deriv
            assert len(pdemod.dG_ddelYf) == 2  # HACK
            assert len(pdemod.dG_ddeldelYf) == 2  # HACK
            assert len(pdemod.bcf) == 2        # HACK
            assert len(pdemod.bcdf) == 2       # HACK
            assert len(pdemod.bcd2f) == 2       # HACK
            self.Gf = pdemod.Gf
            self.dG_dYf = pdemod.dG_dYf
            self.dG_ddelYf = pdemod.dG_ddelYf
            self.dG_ddeldelYf = pdemod.dG_ddeldelYf
            self.bcf = pdemod.bcf
            self.bcdf = pdemod.bcdf
            self.bcd2f = pdemod.bcd2f
            if hasattr(pdemod, 'Yaf'):
                self.Yaf = pdemod.Yaf
            if hasattr(pdemod, 'delYaf'):
                self.delYaf = pdemod.delYaf
            if hasattr(pdemod, 'deldelYaf'):
                self.deldelYaf = pdemod.deldelYaf

    def __str__(self):
        s = ''
        s += 'PDE2DIFF1D:\n'
        s += "name = %s\n" % self.name
        s += "Gf = %s\n" % \
            (getsource(self.Gf).rstrip() if self.Gf else None)
        s += "dG_dYf = %s\n" % \
            (getsource(self.dG_dYf).rstrip() if self.dG_dYf else None)
        if self.dG_ddelYf:
            for (i, f) in enumerate(self.dG_ddelYf):
                s += "dG_ddelYf[%d] = %s\n" % (i, getsource(f).rstrip())
        else:
            s += "dG_ddelYf = %s\n" % None
        if self.dG_ddeldelYf:
            for i in range(len(self.dG_ddeldelYf)):
                for (j, f) in enumerate(self.dG_ddeldelYf[i]):
                    s += "dG_ddeldelYf[%d][%d] = %s\n" % \
                        (i, j, getsource(f).rstrip())
        else:
            s += "dG_ddeldelYf = %s\n" % None
        if self.bcf:
            for i in range(len(self.bcf)):
                for (j, f) in enumerate(self.bcf[i]):
                    s += "bcf[%d][%d] = %s\n" % (i, j, getsource(f).rstrip())
        else:
            s += "bcf = %s\n" % None
        if self.bcdf:
            for i in range(len(self.bcdf)):
                for (j, f) in enumerate(self.bcdf[i]):
                    s += "bcdf[%d][%d] = %s\n" % (i, j, getsource(f).rstrip())
        else:
            s += "bcdf = %s\n" % None
        if self.bcd2f:
            for i in range(len(self.bcd2f)):
                for (j, f) in enumerate(self.bcd2f[i]):
                    s += "bcdf[%d][%d] = %s\n" % (i, j, getsource(f).rstrip())
        else:
            s += "bcd2f = %s\n" % None
        s += "Yaf = %s\n" % (getsource(self.Yaf).rstrip()
                             if self.Yaf else None)
        if self.delYaf:
            for (i, f) in enumerate(self.delYaf):
                s += "delYaf[%d] = %s\n" % (i, getsource(f).rstrip())
        else:
            s += "delYaf = %s\n" % None
        if self.deldelYaf:
            for i in range(len(self.deldelYaf)):
                for (j, f) in enumerate(self.deldelYaf[i]):
                    s += "bcdf[%d][%d] = %s\n" % (i, j, getsource(f).rstrip())
        else:
            s += "deldelYaf = %s\n" % None
        return s.rstrip()  # Strip trailing newline if any.


if __name__ == '__main__':
    pde2diff1d = PDE2DIFF1D('diff1d_0')
    print(pde2diff1d)
