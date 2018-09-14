"""
Base class for 2-D 1st-order partial differential equation initial value
problems

This module provides the base functionality for all 1st-order partial
differential equation initial value problem objects used in the nnode software.

Example:
    Create an empty PDE1IVP object.
        pde1ivp = PDE1IVP()

Attributes:
    None

Methods:
    None

Todo:
    * Expand base functionality.
"""


from importlib import import_module
from inspect import getsource

from pde1 import PDE1


class PDE1IVP(PDE1):
    """Base class for 2-D 1st-order PDE IVPs"""

    def __init__(self, diffeqmod=None):
        self.name = None
        self.Gf = None
        self.dG_dYf = None
        self.dG_ddelYf = None
        self.bcf = None
        self.bcdf = None
        self.Yaf = None
        self.delYaf = None
        if diffeqmod:
            self.name = diffeqmod
            pdemod = import_module(diffeqmod)
            assert pdemod.Gf          # Function for the PDE as a whole
            assert pdemod.dG_dYf      # dG/dY   Y=Y(x,y)
            assert pdemod.dG_ddelYf   # dG/dgrad(Y)
            assert pdemod.bcf         # Array of initial condition functions
            assert pdemod.bcdf        # Array of initial condition derivatives
            assert pdemod.Yaf         # Analytical solution
            assert pdemod.delYaf      # Analytical gradient
            assert len(pdemod.dG_ddelYf) == 2  # HACK
            assert len(pdemod.bcf) == 2        # HACK
            assert len(pdemod.bcdf) == 2       # HACK
            assert len(pdemod.delYaf) == 2     # HACK
            self.Gf = pdemod.Gf
            self.dG_dYf = pdemod.dG_dYf
            self.dG_ddelYf = pdemod.dG_ddelYf
            self.bcf = pdemod.bcf
            self.bcdf = pdemod.bcdf
            if pdemod.Yaf:
                self.Yaf = pdemod.Yaf
            if pdemod.delYaf:
                self.delYaf = pdemod.delYaf

    def __str__(self):
        s = ''
        s += 'PDE1IVP:\n'
        s += "name = %s\n" % self.name
        s += "Gf = %s\n" % (getsource(self.Gf).rstrip() if self.Gf else None)
        s += "dG_dYf = %s\n" % (getsource(self.dG_dYf).rstrip()
                                if self.dG_dYf else None)
        for i in range(2):
            s += "dG_ddelYf[%d] = %s\n" % \
                (i, getsource(self.dG_ddelYf[i]).rstrip()
                 if self.dG_ddelYf[i] else None)
        for i in range(2):
            s += "bcf[%d] = %s\n" % \
                (i, getsource(self.bcf[i]).rstrip() if self.bcf[i] else None)
        for i in range(2):
            s += "bcdf[%d] = %s\n" % \
                (i, getsource(self.bcdf[i]).rstrip() if self.bcdf[i] else None)
        s += "Yaf = %s" % (getsource(self.Yaf).rstrip() if self.Yaf else None)
        for i in range(2):
            s += "delYaf[%d] = %s\n" % \
                (i, getsource(self.delYaf[i]).rstrip()
                 if self.delYaf[i] else None)
        return s.rstrip()  # Strip trailing newline if any.


if __name__ == '__main__':
    pde1ivp = PDE1IVP('pde00')
    print(pde1ivp)
