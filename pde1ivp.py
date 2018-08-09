# N.B. FOR 2-D PDEs ONLY!

from importlib import import_module

from pde1 import PDE1

class PDE1IVP(PDE1):

    def __init__(self, diffeqmod = None):
        """Create a PDE1IVP object from a Python module."""
        self.Gf = None
        self.dG_dYf = None
        self.dG_ddelYf = None
        self.bcf = None
        self.bcdf = None
        self.Yaf = None
        self.delYaf = None
        if diffeqmod:
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
            self.Yaf = pdemod.Yaf
            self.delYaf = pdemod.delYaf

if __name__ == '__main__':
    pde1ivp = PDE1IVP('pde00')
    print(pde1ivp)
