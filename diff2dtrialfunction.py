################################################################################
"""
Diff2DTrialFunction - Class implementing the trial function for 2-D diffusion
problems

The trial function takes the form:

Yt(x,y,t) = A(x,y,t) + P(x,y,t)N(x,y,t,p)

where:

A(x,y,t) = boundary condition function that reduces to BC at boundaries
P(x,y,t) = network coefficient function that vanishes at boundaries
N(x,y,t,p) = scalar output of neural network with parameter vector p

Example:
    Create a default Diff2DTrialFunction object
        Yt = Diff2DTrialFunction()

Notes:
    * Function names end in 'f'.

Attributes:
    None

Methods:
    None

Todo:
    * Expand basic functionality
"""


import numpy as np


class Diff2DTrialFunction():
    """Trial function for 2D diffusion problems."""


    # Public methods

    def __init__(self, bcf, delbcf, del2bcf):
        """Constructor"""
        self.bcf = bcf
        self.delbcf = delbcf
        self.del2bcf = del2bcf

    def Ytf(self, xyt, N):
        """Trial function"""
        A = self.Af(xyt)
        P = self.Pf(xyt)
        Yt = A + P*N
        return Yt

    def delYtf(self, xyt, N, delN):
        """Trial function gradient"""
        (x, y, t) = xyt
        (dN_dx, dN_dy, dN_dt) = delN
        (dA_dx, dA_dy, dA_dt) = self.delAf(xyt)
        P = self.Pf(xyt)
        (dP_dx, dP_dy, dP_dt) = self.delPf(xyt)
        dYt_dx = dA_dx + P*dN_dx + dP_dx*N
        dYt_dy = dA_dy + P*dN_dy + dP_dy*N
        dYt_dt = dA_dt + P*dN_dt + dP_dt*N
        delYt = [dYt_dx, dYt_dy, dYt_dt]
        return delYt

    def del2Ytf(self, xyt, N, delN, del2N):
        """Trial function Laplacian"""
        (x, y, t) = xyt
        (dN_dx, dN_dy, dN_dt) = delN
        (d2N_dx2, d2N_dy2, d2N_dt2) = del2N
        (d2A_dx2, d2A_dy2, d2A_dt2) = self.del2Af(xyt)
        P = self.Pf(xyt)
        (dP_dx, dP_dy, dP_dt) = self.delPf(xyt)
        (d2P_dx2, d2P_dy2, d2P_dt2) = self.del2Pf(xyt)
        d2Yt_dx2 = d2A_dx2 + P*d2N_dx2 + 2*dP_dx*dN_dx + d2P_dx2*N
        d2Yt_dy2 = d2A_dy2 + P*d2N_dy2 + 2*dP_dy*dN_dy + d2P_dy2*N
        d2Yt_dt2 = d2A_dt2 + P*d2N_dt2 + 2*dP_dt*dN_dt + d2P_dt2*N
        del2Yt = [d2Yt_dx2, d2Yt_dy2, d2Yt_dt2]
        return del2Yt

    def Af(self, xyt):
        """Boundary condition function"""
        (x, y, t) = xyt
        (cx1, cx2, cy1, cy2, ct1) = self.cf(xyt)
        A = cx1*x + cx2*x**2 + cy1*y + cy2*y**2 + ct1*t
        return A

    def delAf(self, xyt):
        """Gradient of boundary condition function"""
        (x, y, t) = xyt
        (cx1, cx2, cy1, cy2, ct1) = self.cf(xyt)
        ((dcx1_dx, dcx1_dy, dcx1_dt),
         (dcx2_dx, dcx2_dy, dcx2_dt),
         (dcy1_dx, dcy1_dy, dcy1_dt),
         (dcy2_dx, dcy2_dy, dcy2_dt),
         (dct1_dx, dct1_dy, dct1_dt),
        ) = self.delcf(xyt)
        dA_dx = (cx1 + dcx1_dx*x) + (cx2*2*x + dcx2_dx*x**2) + (dcy1_dx*y) + (dcy2_dx*y**2) + (dct1_dx*t)
        dA_dy = (dcx1_dy*x) + (dcx2_dy*x**2) + (cy1 + dcy1_dy*y) + (cy2*2*y + dcy2_dy*y**2) + (dct1_dy*t)
        dA_dt = (dcx1_dt*x) + (dcx2_dt*x**2) + (dcy1_dt*y) + (dcy2_dt*y**2) + (ct1 + dct1_dt*t)
        delA = [dA_dx, dA_dy, dA_dt]
        return delA

    def del2Af(self, xyt):
        (x, y, t) = xyt
        (cx1, cx2, cy1, cy2, ct1) = self.cf(xyt)
        ((dcx1_dx, dcx1_dy, dcx1_dt),
         (dcx2_dx, dcx2_dy, dcx2_dt),
         (dcy1_dx, dcy1_dy, dcy1_dt),
         (dcy2_dx, dcy2_dy, dcy2_dt),
         (dct1_dx, dct1_dy, dct1_dt),
        ) = self.delcf(xyt)
        ((d2cx1_dx2, d2cx1_dy2, d2cx1_dt2),
         (d2cx2_dx2, d2cx2_dy2, d2cx2_dt2),
         (d2cy1_dx2, d2cy1_dy2, d2cy1_dt2),
         (d2cy2_dx2, d2cy2_dy2, d2cy2_dt2),
         (d2ct1_dx2, d2ct1_dy2, d2ct1_dt2),
        ) = self.del2cf(xyt)

        d2A_dx2 = (2*dcx1_dx + d2cx1_dx2*x) + (cx2*2 + 2*dcx2_dx*2*x + d2cx2_dx2*x**2) + (d2cy1_dx2*y) + (d2cy2_dx2*y**2) + (d2ct1_dx2*t)
        d2A_dy2 = (d2cx1_dy2*x) + (d2cx2_dy2*x**2) + (2*dcy1_dy + d2cy1_dy2*y) + (cy2*2 + 2*dcy2_dy*2*y + d2cy2_dy2*y**2) + (d2ct1_dy2*t)
        d2A_dt2 = (d2cx1_dt2*x) + (d2cx2_dt2*x**2) + (d2cy1_dt2*y) + (d2cy2_dt2*y**2) + (2*dct1_dt + d2ct1_dt2*t)
        del2A = [d2A_dx2, d2A_dy2, d2A_dt2]
        return del2A

    def cf(self, xyt):
        """Compute the coefficient vector for the boundary condition function"""
        (x, y, t) = xyt
        ((f0f, f1f), (g0f, g1f), (Y0f, Y1f)) = self.bcf
        f0 = f0f(xyt); f1 = f1f(xyt)
        g0 = g0f(xyt); g1 = g1f(xyt)
        Y0 = Y0f(xyt); Y1 = Y1f(xyt)
        cx1 = (f0*(2*x**2 - 1) - 2*f1*x**2 + g0 + Y0)/(2*x*(1 - x))
        cx2 = (f0*(1 - 2*x) + 2*f1*x - g0 - Y0)/(2*x*(1 - x))
        cy1 = (f0 + g0*(2*y**2 - 1) - 2*g1*y**2 + Y0)/(2*y*(1 - y))
        cy2 = (-f0 + g0*(1 - 2*y) + 2*g1*y - Y0)/(2*y*(1 - y))
        ct1 = (f0 + g0 - Y0)/(2*t)
        c = [cx1, cx2, cy1, cy2, ct1]
        return c

    def delcf(self, xyt):
        """Compute the gradients of each coefficient."""
        (x, y, t) = xyt
        ((f0f, f1f), (g0f, g1f), (Y0f, Y1f)) = self.bcf
        (((df0_dxf, df0_dyf, df0_dtf), (df1_dxf, df1_dyf, df1_dtf)),
         ((dg0_dxf, dg0_dyf, dg0_dtf), (dg1_dxf, dg1_dyf, dg1_dtf)),
         ((dY0_dxf, dY0_dyf, dY0_dtf), (dY1_dxf, dY1_dyf, dY1_dtf))) = self.delbcf
        f0 = f0f(xyt); f1 = f1f(xyt)
        g0 = g0f(xyt); g1 = g1f(xyt)
        Y0 = Y0f(xyt); Y1 = Y1f(xyt)
        df0_dx = df0_dxf(xyt); df0_dy = df0_dyf(xyt); df0_dt = df0_dtf(xyt)
        df1_dx = df1_dxf(xyt); df1_dy = df1_dyf(xyt); df1_dt = df1_dtf(xyt)
        dg0_dx = dg0_dxf(xyt); dg0_dy = dg0_dyf(xyt); dg0_dt = dg0_dtf(xyt)
        dg1_dx = dg1_dxf(xyt); dg1_dy = dg1_dyf(xyt); dg1_dt = dg1_dtf(xyt)
        dY0_dx = dY0_dxf(xyt); dY0_dy = dY0_dyf(xyt); dY0_dt = dY0_dtf(xyt)
        dY1_dx = dY1_dxf(xyt); dY1_dy = dY1_dyf(xyt); dY1_dt = dY1_dtf(xyt)

        dcx1_dx = -((-1 - 2*(-1 + x)*x)*f0 + g0 + Y0 +
                    x*(2*x*f1 - 2*g0 - 2*Y0 + (-1 + x)*
                       ((-1 + 2*x**2)*df0_dx - 2*x**2*df1_dx + dg0_dx +
                        dY0_dx)))/ \
                  (2*(-1 + x)**2*x**2)
        dcx1_dy = -((-1 + 2*x**2)*df0_dy - 2*x**2*df1_dy + dg0_dy + dY0_dy)/ \
                  (2*(-1 + x)*x)
        dcx1_dt = -((-1 + 2*x**2)*df0_dt - 2*x**2*df1_dt + dg0_dt + dY0_dt)/ \
                  (2*(-1 + x)*x)

        dcx2_dx = ((-1 - 2*(-1 + x)*x)*f0 + g0 + Y0 +
                   x*(2*x*f1 - 2*g0 - 2*Y0 + (-1 + x)*
                      ((-1 + 2*x)*df0_dx - 2*x*df1_dx + dg0_dx + dY0_dx)))/ \
                  (2*(-1 + x)**2*x**2)
        dcx2_dy = ((-1 + 2*x)*df0_dy - 2*x*df1_dy + dg0_dy + dY0_dy)/ \
                  (2*(-1 + x)*x)
        dcx2_dt = ((-1 + 2*x)*df0_dt - 2*x*df1_dt + dg0_dt + dY0_dt)/ \
                  (2*(-1 + x)*x)

        dcy1_dx = -(df0_dx + (-1 + 2*y**2)*dg0_dx - 2*y**2*dg1_dx + dY0_dx)/ \
                  (2*(-1 + y)*y)
        dcy1_dy = ((-1 + 2*y)*f0 + (1 + 2*(-1 + y)*y)*g0 - Y0 +
                   y*(-2*y*g1 + 2*Y0 - (-1 + y)*
                      (df0_dy + (-1 + 2*y**2)*dg0_dy - 2*y**2*dg1_dy +
                       dY0_dy)))/ \
                  (2*(-1 + y)**2*y**2)
        dcy1_dt = -(df0_dt + (-1 + 2*y**2)*dg0_dt - 2*y**2*dg1_dt + dY0_dt)/ \
                  (2*(-1 + y)*y)

        dcy2_dx = (df0_dx + (-1 + 2*y)*dg0_dx - 2*y*dg1_dx + dY0_dx)/ \
                  (2*(-1 + y)*y)
        dcy2_dy = ((1 - 2*y)*f0 + (-1 - 2*(-1 + y)*y)*g0 + Y0 +
                   y*(2*y*g1 - 2*Y0 + (-1 + y)*
                      (df0_dy + (-1 + 2*y)*dg0_dy - 2*y*dg1_dy + dY0_dy)))/ \
                  (2*(-1 + y)**2*y**2)
        dcy2_dt = (df0_dt + (-1 + 2*y)*dg0_dt - 2*y*dg1_dt + dY0_dt)/ \
                  (2*(-1 + y)*y)

        dct1_dx = (df0_dx + dg0_dx - dY0_dx)/(2*t)
        dct1_dy = (df0_dy + dg0_dy - dY0_dy)/(2*t)
        dct1_dt = (-f0 - g0 + Y0 + t*(df0_dt + dg0_dt - dY0_dt))/(2*t**2)

        delc = [
            [dcx1_dx, dcx1_dy, dcx1_dt],
            [dcx2_dx, dcx2_dy, dcx2_dt],
            [dcy1_dx, dcy1_dy, dcy1_dt],
            [dcy2_dx, dcy2_dy, dcy2_dt],
            [dct1_dx, dct1_dy, dct1_dt]
            ]
        return delc

    def del2cf(self, xyt):
        """Compute the Laplacians of each coefficient."""
        (x, y, t) = xyt
        ((f0f, f1f), (g0f, g1f), (Y0f, Y1f)) = self.bcf
        (((df0_dxf, df0_dyf, df0_dtf), (df1_dxf, df1_dyf, df1_dtf)),
         ((dg0_dxf, dg0_dyf, dg0_dtf), (dg1_dxf, dg1_dyf, dg1_dtf)),
         ((dY0_dxf, dY0_dyf, dY0_dtf), (dY1_dxf, dY1_dyf, dY1_dtf))) = self.delbcf
        (((d2f0_dx2f, d2f0_dy2f, d2f0_dt2f), (d2f1_dx2f, d2f1_dy2f, d2f1_dt2f)),
         ((d2g0_dx2f, d2g0_dy2f, d2g0_dt2f), (d2g1_dx2f, d2g1_dy2f, d2g1_dt2f)),
         ((d2Y0_dx2f, d2Y0_dy2f, d2Y0_dt2f), (d2Y1_dx2f, d2Y1_dy2f, d2Y1_dt2f))) = self.del2bcf

        f0 = f0f(xyt); f1 = f1f(xyt)
        g0 = g0f(xyt); g1 = g1f(xyt)
        Y0 = Y0f(xyt); Y1 = Y1f(xyt)

        df0_dx = df0_dxf(xyt); df0_dy = df0_dyf(xyt); df0_dt = df0_dtf(xyt)
        df1_dx = df1_dxf(xyt); df1_dy = df1_dyf(xyt); df1_dt = df1_dtf(xyt)
        dg0_dx = dg0_dxf(xyt); dg0_dy = dg0_dyf(xyt); dg0_dt = dg0_dtf(xyt)
        dg1_dx = dg1_dxf(xyt); dg1_dy = dg1_dyf(xyt); dg1_dt = dg1_dtf(xyt)
        dY0_dx = dY0_dxf(xyt); dY0_dy = dY0_dyf(xyt); dY0_dt = dY0_dtf(xyt)
        dY1_dx = dY1_dxf(xyt); dY1_dy = dY1_dyf(xyt); dY1_dt = dY1_dtf(xyt)

        d2f0_dx2 = d2f0_dx2f(xyt); d2f0_dy2 = d2f0_dy2f(xyt); d2f0_dt2 = d2f0_dt2f(xyt)
        d2f1_dx2 = d2f1_dx2f(xyt); d2f1_dy2 = d2f1_dy2f(xyt); d2f1_dt2 = d2f1_dt2f(xyt)
        d2g0_dx2 = d2g0_dx2f(xyt); d2g0_dy2 = d2g0_dy2f(xyt); d2g0_dt2 = d2g0_dt2f(xyt)
        d2g1_dx2 = d2g1_dx2f(xyt); d2g1_dy2 = d2g1_dy2f(xyt); d2g1_dt2 = d2g1_dt2f(xyt)
        d2Y0_dx2 = d2Y0_dx2f(xyt); d2Y0_dy2 = d2Y0_dy2f(xyt); d2Y0_dt2 = d2Y0_dt2f(xyt)
        d2Y1_dx2 = d2Y1_dx2f(xyt); d2Y1_dy2 = d2Y1_dy2f(xyt); d2Y1_dt2 = d2Y1_dt2f(xyt)

        d2cx1_dx2 = ((2 - 2*x*(3 + x*(-3 + 2*x)))*f0 + 4*x**3*f1 -
                     2*(g0 + Y0) + (-1 + x)*x*
                     (-6*g0 - 6*Y0 + (2 + 4*(-1 + x)*x)*df0_dx -
                     2*(dg0_dx + dY0_dx) +
                     x*(-4*x*df1_dx + 4*dg0_dx + 4*dY0_dx -
                     (-1 + x)*((-1 + 2*x**2)*d2f0_dx2 - 2*x**2*d2f1_dx2 +
                               d2g0_dx2 + d2Y0_dx2))))/ \
                    (2*(-1 + x)**3*x**3)
        d2cx1_dy2 = -((-1 + 2*x**2)*d2f0_dx2 - 2*x**2*d2f1_dx2 +
                      d2g0_dx2 + d2Y0_dx2)/ \
                    (2*(-1 + x)*x)
        d2cx1_dt2 = -((-1 + 2*x**2)*d2f0_dt2 - 2*x**2*d2f1_dt2 +
                      d2g0_dt2 + d2Y0_dt2)/ \
                    (2*(-1 + x)*x)
        d2cx2_dx2 = (2*(-1 + 2*x)*(1 + (-1 + x)*x)*f0 - 4*x**3*f1 +
                     2*(g0 + Y0) + (-1 + x)*x*
                     (6*g0 + 6*Y0 + (-2 - 4*(-1 + x)*x)*df0_dx +
                      2*(dg0_dx + dY0_dx) +
                      x*(4*x*df1_dx - 4*dg0_dx - 4*dY0_dx +
                      (-1 + x)*((-1 + 2*x)*d2f0_dx2 - 2*x*d2f1_dx2 +
                                d2g0_dx2 + d2Y0_dx2))))/ \
                    (2*(-1 + x)**3*x**3)
        d2cx2_dy2 = ((-1 + 2*x)*d2f0_dy2 - 2*x*d2f1_dy2 + d2g0_dy2 + d2Y0_dy2)/ \
                    (2*(-1 + x)*x)
        d2cx2_dt2 = ((-1 + 2*x)*d2f0_dt2 - 2*x*d2f1_dt2 + d2g0_dt2 + d2Y0_dt2)/ \
                    (2*(-1 + x)*x)
        d2cy1_dx2 = -(d2f0_dx2 + (-1 + 2*y**2)*d2g0_dx2 - 2*y**2*d2g1_dx2 + d2Y0_dx2)/ \
                    (2*(-1 + y)*y)
        d2cy1_dy2 = ((-2 - 6*(-1 + y)*y)*f0 + (2 - 2*y*(3 + y*(-3 + 2*y)))*g0 +
                     4*y**3*g1 - 2*Y0 + (-1 + y)*y*
                     (-6*Y0 + (-2 + 4*y)*df0_dy + (2 + 4*(-1 + y)*y)*dg0_dy -
                     2*dY0_dy + y*(-4*y*dg1_dy + 4*dY0_dy - (-1 + y)*
                     (d2f0_dy2 + (-1 + 2*y**2)*d2g0_dy2 - 2*y**2*d2g1_dy2 +
                      d2Y0_dy2))))/ \
                    (2*(-1 + y)**3*y**3)
        d2cy1_dt2 = -(d2f0_dt2 + (-1 + 2*y**2)*d2g0_dt2 - 2*y**2*d2g1_dt2 + d2Y0_dt2)/ \
                    (2*(-1 + y)*y)
        d2cy2_dx2 = (d2f0_dx2 + (-1 + 2*y)*d2g0_dx2 - 2*y*d2g1_dx2 + d2Y0_dx2)/ \
                    (2*(-1 + y)*y)
        d2cy2_dy2 = ((2 + 6*(-1 + y)*y)*f0 + 2*(-1 + 2*y)*(1 + (-1 + y)*y)*g0 -
                     4*y**3*g1 + 2*Y0 + (-1 + y)*y*
                     (6*Y0 + (2 - 4*y)*df0_dy + (-2 - 4*(-1 + y)*y)*dg0_dy +
                      2*dY0_dy + y*(4*y*dg1_dy - 4*dY0_dy + (-1 + y)*
                                    (d2f0_dy2 + (-1 + 2*y)*d2g0_dy2 - 2*y*d2g1_dy2 +
                                     d2Y0_dy2))))/ \
                    (2*(-1 + y)**3*y**3)
        d2cy2_dt2 = (d2f0_dt2 + (-1 + 2*y)*d2g0_dt2 - 2*y*d2g1_dt2 + d2Y0_dt2)/ \
                    (2*(-1 + y)*y)
        d2ct1_dx2 = (d2f0_dx2 + d2g0_dx2 - d2Y0_dx2)/(2*t)
        d2ct1_dy2 = (d2f0_dy2 + d2g0_dy2 - d2Y0_dy2)/(2*t)
        d2ct1_dt2 = (2*f0 + 2*g0 - 2*Y0 + t*
                     (-2*df0_dt - 2*dg0_dt + 2*dY0_dt + t*
                      (d2f0_dt2 + d2g0_dt2 - d2Y0_dt2)))/ \
                    (2*t**3)
        
        del2c = [
            [d2cx1_dx2, d2cx1_dy2, d2cx1_dt2],
            [d2cx2_dx2, d2cx2_dy2, d2cx2_dt2],
            [d2cy1_dx2, d2cy1_dy2, d2cy1_dt2],
            [d2cy2_dx2, d2cy2_dy2, d2cy2_dt2],
            [d2ct1_dx2, d2ct1_dy2, d2ct1_dt2],
            ]
        return del2c

    def Pf(self, xyt):
        """Network coefficient function for 2D diffusion problems"""
        (x, y, t) = xyt
        P = x*(1 - x)*y*(1 - y)*t
        return P

    def delPf(self, xyt):
        """Network coefficient function gradient"""
        (x, y, t) = xyt
        dP_dx = (1 - 2*x)*y*(1 - y)*t
        dP_dy = x*(1 - x)*(1 - 2*y)*t
        dP_dt = x*(1 - x)*y*(1 - y)
        delP = [dP_dx, dP_dy, dP_dt]
        return delP

    def del2Pf(self, xyt):
        """Network coefficient function Laplacian"""
        (x, y, t) = xyt
        d2P_dx2 = -2*y*(1 - y)*t
        d2P_dy2 = -2*x*(1 - x)*t
        d2P_dt2 = 0
        del2P = [d2P_dx2, d2P_dy2, d2P_dt2]
        return del2P

#################


# Self-test code

from math import pi, sin, cos

if __name__ == '__main__':

    # Test boundary conditions
    f0f = lambda xyt: 0
    f1f = lambda xyt: 0
    g0f = lambda xyt: 0
    g1f = lambda xyt: 0
    Y0f = lambda xyt: sin(pi*xyt[0])*sin(pi*xyt[1])/2
    Y1f = lambda xyt: None
    bcf = [[f0f, f1f],
           [g0f, g1f],
           [Y0f, Y1f]]

    # Test BC gradient
    df0_dxf = lambda xyt: 0
    df0_dyf = lambda xyt: 0
    df0_dtf = lambda xyt: 0
    df1_dxf = lambda xyt: 0
    df1_dyf = lambda xyt: 0
    df1_dtf = lambda xyt: 0
    dg0_dxf = lambda xyt: 0
    dg0_dyf = lambda xyt: 0
    dg0_dtf = lambda xyt: 0
    dg1_dxf = lambda xyt: 0
    dg1_dyf = lambda xyt: 0
    dg1_dtf = lambda xyt: 0
    dY0_dxf = lambda xyt: pi*cos(pi*xyt[0])*sin(pi*xyt[1])/2
    dY0_dyf = lambda xyt: pi*sin(pi*xyt[0])*cos(pi*xyt[1])/2
    dY0_dtf = lambda xyt: 0
    dY1_dxf = lambda xyt: None
    dY1_dyf = lambda xyt: None
    dY1_dtf = lambda xyt: None
    delbcf = [[[df0_dxf, df0_dyf, df0_dtf], [df1_dxf, df1_dyf, df1_dtf]],
              [[dg0_dxf, dg0_dyf, dg0_dtf], [dg1_dxf, dg1_dyf, dg1_dtf]],
              [[dY0_dxf, dY0_dyf, dY0_dtf], [dY1_dxf, dY1_dyf, dY1_dtf]]]

    # Test BC Laplacian
    d2f0_dx2f = lambda xyt: 0
    d2f0_dy2f = lambda xyt: 0
    d2f0_dt2f = lambda xyt: 0
    d2f1_dx2f = lambda xyt: 0
    d2f1_dy2f = lambda xyt: 0
    d2f1_dt2f = lambda xyt: 0
    d2g0_dx2f = lambda xyt: 0
    d2g0_dy2f = lambda xyt: 0
    d2g0_dt2f = lambda xyt: 0
    d2g1_dx2f = lambda xyt: 0
    d2g1_dy2f = lambda xyt: 0
    d2g1_dt2f = lambda xyt: 0
    d2Y0_dx2f = lambda xyt: -pi**2*sin(pi*xyt[0])*sin(pi*xyt[1])/2
    d2Y0_dy2f = lambda xyt: -pi**2*sin(pi*xyt[0])*sin(pi*xyt[1])/2
    d2Y0_dt2f = lambda xyt: 0
    d2Y1_dx2f = lambda xyt: None
    d2Y1_dy2f = lambda xyt: None
    d2Y1_dt2f = lambda xyt: None
    del2bcf = [[[d2f0_dx2f, d2f0_dy2f, d2f0_dt2f], [d2f1_dx2f, d2f1_dy2f, d2f1_dt2f]],
               [[d2g0_dx2f, d2g0_dy2f, d2g0_dt2f], [d2g1_dx2f, d2g1_dy2f, d2g1_dt2f]],
               [[d2Y0_dx2f, d2Y0_dy2f, d2Y0_dt2f], [d2Y1_dx2f, d2Y1_dy2f, d2Y1_dt2f]]]

    # Reference values for tests.
    bc_ref = [[0, 0],
              [0, 0],
              [0.475528, None]]
    delbc_ref = [[[0, 0, 0], [0, 0, 0]],
                 [[0, 0, 0], [0, 0, 0]],
                 [[0.485403, 0, 0], [None, None, None]]]
    del2bc_ref = [[[0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0]],
                  [[-4.69328, -4.69328, 0], [None, None, None]]]
    c_ref = [0.990684, -0.9906849, 0.951057,
             -0.951057, -0.396274]
    delc_ref = [[0.185686, 0, 0],
                [-0.185686, 0, 0],
                [0.970806, 0, 0],
                [-0.970806, 0, 0],
                [-0.404502, 0, .660456]]
    del2c_ref = [[-1.83144, -9.77766, 0],
                 [1.83144, 9.77766, 0],
                 [-9.38655, -1.7781, 0],
                 [9.38655, 1.7781, 0],
                 [3.91106, 3.91106, -2.20152]]
    A_ref = 0.237764
    delA_ref = [0.242701, 0, 0]
    del2A_ref = [-2.34664, -2.34664, 0]
    P_ref = 0.036
    delP_ref = [0.03, 0.0, 0.06]
    del2P_ref = [-0.3, -0.288, 0]
    Yt_ref = 0.242192
    delYt_ref = [0.249991, 0.0072, 0.01818]
    del2Yt_ref = [-2.37358, -2.37414, 0.04788]

    # # Additional test variables.
    N_ref = 0.123
    delN_ref = [0.1, 0.2, 0.3]
    del2N_ref = [0.11, 0.22, 0.33]

    # Test all functions near the center of the domain.
    xyt = [0.4, 0.5, 0.6]

    # Create a new trial function object.
    tf = Diff2DTrialFunction(bcf, delbcf, del2bcf)

    print("Testing boundary conditions.")
    for i in range(len(tf.bcf)):
        for (j, f) in enumerate(tf.bcf[i]):
            bc = f(xyt)
            if ((bc_ref[i][j] is not None and not np.isclose(bc, bc_ref[i][j]))
                or (bc_ref[i][j] is None and bc is not None)):
                print("ERROR: bc[%d][%d] = %s, vs ref %s" % (i, j, bc, bc_ref[i][j]))

    print("Testing boundary condition gradients.")
    for i in range(len(tf.delbcf)):
        for j in range(len(tf.delbcf[i])):
            for (k, f) in enumerate(tf.delbcf[i][j]):
                delbc = f(xyt)
                if ((delbc_ref[i][j][k] is not None and not np.isclose(delbc, delbc_ref[i][j][k]))
                    or (delbc_ref[i][j][k] is None and delbc is not None)):
                    print("ERROR: delbc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, delbc, delbc_ref[i][j][k]))

    print("Testing boundary condition Laplacians.")
    for i in range(len(tf.del2bcf)):
        for j in range(len(tf.del2bcf[i])):
            for (k, f) in enumerate(tf.del2bcf[i][j]):
                del2bc = f(xyt)
                if ((del2bc_ref[i][j][k] is not None and not np.isclose(del2bc, del2bc_ref[i][j][k]))
                    or (del2bc_ref[i][j][k] is None and del2bc is not None)):
                    print("ERROR: del2bc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, del2bc, del2bc_ref[i][j][k]))

    print("Testing coefficients.")
    c = tf.cf(xyt)
    for (i, cc) in enumerate(c):
        if not np.isclose(cc, c_ref[i]):
            print("ERROR: c[%d] = %s, vs ref %s" % (i, cc, c_ref[i]))

    print("Testing coefficient gradients.")
    delc = tf.delcf(xyt)
    for i in range(len(c)):
        for (j, delci) in enumerate(delc[i]):
            if not np.isclose(delci, delc_ref[i][j]):
                print("ERROR: delc[%d][%d] = %s, vs ref %s" % (i, j, delci, delc_ref[i][j]))

    print("Testing coefficient Laplacians.")
    del2c = tf.del2cf(xyt)
    for i in range(len(del2c)):
        for (j, del2ci) in enumerate(del2c[i]):
            if not np.isclose(del2ci, del2c_ref[i][j]):
                print("ERROR: del2c[%d][%d] = %s, vs ref %s" % (i, j, del2ci, del2c_ref[i][j]))

    print("Testing boundary condition function.")
    A = tf.Af(xyt)
    if not np.isclose(A, A_ref):
        print("ERROR: A = %s, vs ref %s" % (A, A_ref))

    print("Testing boundary condition function gradient.")
    delA = tf.delAf(xyt)
    for (i, delAi) in enumerate(delA):
        if not np.isclose(delAi, delA_ref[i]):
            print("ERROR: delA[%d] = %s, vs ref %s" % (i, delAi, delA_ref[i]))

    print("Testing boundary condition function Laplacian.")
    del2A = tf.del2Af(xyt)
    for (i, del2Ai) in enumerate(del2A):
        if not np.isclose(del2Ai, del2A_ref[i]):
            print("ERROR: del2A[%d] = %s, vs ref %s" % (i, del2Ai, del2A_ref[i]))

    print("Testing network coefficient function.")
    P = tf.Pf(xyt)
    if not np.isclose(P, P_ref):
        print("ERROR: P = %s, vs ref %s" % (P, P_ref))

    print("Testing network coefficient function gradient.")
    delP = tf.delPf(xyt)
    for (i, delPi) in enumerate(delP):
        if not np.isclose(delPi, delP_ref[i]):
            print("ERROR: delP[%d] = %s, vs ref %s" % (delPi, delP_ref[i]))

    print("Testing network coefficient function Laplacian.")
    del2P = tf.del2Pf(xyt)
    for (i, del2Pi) in enumerate(del2P):
        if not np.isclose(del2Pi, del2P_ref[i]):
            print("ERROR: del2P[%d] = %s, vs ref %s" % (del2Pi, del2P_ref[i]))

    print("Testing trial function.")
    Yt = tf.Ytf(xyt, N_ref)
    if not np.isclose(Yt, Yt_ref):
        print("ERROR: Yt = %s, vs ref %s" % (Yt, Yt_ref))

    print("Testing trial function gradient.")
    delYt = tf.delYtf(xyt, N_ref, delN_ref)
    for (i, delYti) in enumerate(delYt):
        if not np.isclose(delYti, delYt_ref[i]):
            print("ERROR: delYt[%d] = %s, vs ref %s" % (i, delYti, delYt_ref[i]))

    print("Testing trial function Laplacian.")
    del2Yt = tf.del2Ytf(xyt, N_ref, delN_ref, del2N_ref)
    for (i, del2Yti) in enumerate(del2Yt):
        if not np.isclose(del2Yti, del2Yt_ref[i]):
            print("ERROR: del2Yt[%d] = %s, vs ref %s" % (i, del2Yti, del2Yt_ref[i]))
