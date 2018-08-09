# N.B. FOR 2-D PDEs ONLY!

from importlib import import_module

from pde2 import PDE2

class PDE2BVP(PDE2):

    def __init__(self, diffeqmod = None):
        """Create a PDE2BVP object from a Python module."""
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
            pdemod = import_module(diffeqmod)
            assert pdemod.Gf          # Function for the PDE as a whole
            assert pdemod.dG_dYf      # dG/dY   Y=Y(x,y)
            assert pdemod.dG_ddelYf   # dG/dgrad(Y)
            assert pdemod.dG_ddeldelYf  # dG/d(grad(Y)**2)
            assert pdemod.bcf         # Array of initial condition functions
            assert pdemod.bcdf        # Array of initial condition derivatives
            assert pdemod.bcd2f      # Array of initial condition 2nd derivatives
            assert pdemod.Yaf         # Analytical solution
            assert pdemod.delYaf      # Analytical gradient
            assert pdemod.deldelYaf      # Analytical Hessian
            assert len(pdemod.dG_ddelYf) == 2  # HACK
            assert len(pdemod.dG_ddeldelYf) == 2  # HACK
            assert len(pdemod.bcf) == 2        # HACK
            assert len(pdemod.bcdf) == 2       # HACK
            assert len(pdemod.bcd2f) == 2       # HACK
            assert len(pdemod.delYaf) == 2     # HACK
            self.Gf = pdemod.Gf
            self.dG_dYf = pdemod.dG_dYf
            self.dG_ddelYf = pdemod.dG_ddelYf
            self.dG_ddeldelYf = pdemod.dG_ddeldelYf
            self.bcf = pdemod.bcf
            self.bcdf = pdemod.bcdf
            self.bcd2f = pdemod.bcd2f
            self.Yaf = pdemod.Yaf
            self.delYaf = pdemod.delYaf
            self.deldelYaf = pdemod.deldelYaf

if __name__ == '__main__':
    pde2bvp = PDE2BVP('pde02bvp')
    print(pde2bvp)
