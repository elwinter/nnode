"""
PDE2DIFF3D - Base class for 3-D diffusion equations

This module provides the base functionality for all 3-D diffusion equation
objects used in the nnode software.

Example:
    Create an empty PDE2DIFF3D object.
        pde2diff3d = PDE2DIFF3D()
    Create an PDE2DIFF3D object from a Python module.
        pde2diff3d = PDE2DIFF3D(modname)

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


class PDE2DIFF2D(PDE2):

    def __init__(self, diffeqmod=None):
        self.name = None
        self.Gf = None
        self.dG_dYf = None
        self.dG_ddelYf = None
        self.dG_ddel2Yf = None
        self.bcf = None
        self.delbcf = None
        self.del2bcf = None
        self.Yaf = None
        self.delYaf = None
        self.del2Yaf = None
        if diffeqmod:
            self.name = diffeqmod
            pdemod = import_module(diffeqmod)
            assert pdemod.Gf          # Function for the PDE as a whole
            assert pdemod.dG_dYf      # dG/dY   Y=Y(x,y,t)
            assert pdemod.dG_ddelYf   # dG/dgrad(Y)
            assert pdemod.dG_ddel2Yf  # dG/d(grad(Y)grad(Y))
            assert pdemod.bcf         # Array of initial condition functions
            assert pdemod.delbcf        # Array of initial condition gradient functions
            assert pdemod.del2bcf       # Array of initial condition Laplacians
            assert len(pdemod.dG_ddelYf) == 3  # HACK
            assert len(pdemod.dG_ddel2Yf) == 3  # HACK
            assert len(pdemod.bcf) == 3        # HACK
            assert len(pdemod.delbcf) == 3       # HACK
            assert len(pdemod.del2bcf) == 3       # HACK
            self.Gf = pdemod.Gf
            self.dG_dYf = pdemod.dG_dYf
            self.dG_ddelYf = pdemod.dG_ddelYf
            self.dG_ddel2Yf = pdemod.dG_ddel2Yf
            self.bcf = pdemod.bcf
            self.delbcf = pdemod.delbcf
            self.del2bcf = pdemod.del2bcf
            if hasattr(pdemod, 'Yaf'):
                self.Yaf = pdemod.Yaf
            if hasattr(pdemod, 'delYaf'):
                self.delYaf = pdemod.delYaf
            if hasattr(pdemod, 'del2Yaf'):
                self.del2Yaf = pdemod.del2Yaf

    def __str__(self):
        s = ''
        s += 'PDE2DIFF2D:\n'
        s += "name = %s\n" % self.name
        s += "Gf = %s\n" % \
            (getsource(self.Gf).rstrip() if self.Gf else None)
        s += "dG_dYf = %s\n" % \
            (getsource(self.dG_dYf).rstrip() if self.dG_dYf else None)
        for (i, f) in enumerate(self.dG_ddelYf):
                s += "dG_ddelYf[%d] = %s\n" % (i, getsource(f).rstrip())
        for (i, f) in enumerate(self.dG_ddel2Yf):
                s += "dG_ddel2Yf[%d] = %s\n" % (i, getsource(f).rstrip())
        for i in range(len(self.bcf)):
            for (j, f) in enumerate(self.bcf[i]):
                s += "bcf[%d][%d] = %s\n" % (i, j, getsource(f).rstrip())
        for i in range(len(self.delbcf)):
            for j in range(len(self.delbcf[i])):
                for (k, f) in enumerate(self.delbcf[i][j]):
                    s += "delbcf[%d][%d][%d] = %s\n" % (i, j, k, getsource(f).rstrip())
        for i in range(len(self.del2bcf)):
            for j in range(len(self.del2bcf[i])):
                for (k, f) in enumerate(self.del2bcf[i][j]):
                    s += "del2bcf[%d][%d][%d] = %s\n" % (i, j, k, getsource(f).rstrip())
        if self.Yaf:
            s += "Yaf = %s\n" % (getsource(self.Yaf).rstrip()
                                if self.Yaf else None)
        if self.delYaf:
            for (i, f) in enumerate(self.delYaf):
                s += "delYaf[%d] = %s\n" % (i, getsource(f).rstrip())
        if self.del2Yaf:
            for (i, f) in enumerate(self.del2Yaf):
                s += "del2Yaf[%d] = %s\n" % (i, getsource(f).rstrip())
        return s.rstrip()  # Strip trailing newline if any.


if __name__ == '__main__':
    pde2diff2d = PDE2DIFF2D('diff2d_0')
    print(pde2diff2d)
