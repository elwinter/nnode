"""
Diff2DTrialFunction - Class implementing the trial function for 2-D diffusion
problems

The trial function takes the form:

Yt(x,y,t) = A(x,y,t) + P(x,y,t)N(x,y,t,p)

where:

A(x,y,t) = boundary condition function that reduces to BC at boundaries
P(x,y,t) = network coefficient function that vanishes at boundaries
         = x(1-x)y(1-y)t
N(x,y,t,p) = Scalar output of neural network with parameter vector p

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
        """Constructor for Diff2DTrialFunction class"""
        self.bcf = bcf
        self.delbcf = delbcf
        self.del2bcf = del2bcf

    def Ytf(self, xyt, N):
        """Trial function for 2D diffusion problems"""
        A = self.Af(xyt)
        P = self.Pf(xyt)
        Yt = A + P*N
        return Yt

    def delYtf(self, xyt, N, delN):
        """Trial function gradient"""
        delA = self.delAf(xyt)
        P = self.Pf(xyt)
        delP = self.delPf(xyt)
        delYt = [None, None, None]
        delYt[0] = delA[0] + P*delN[0] + delP[0]*N
        delYt[1] = delA[1] + P*delN[1] + delP[1]*N
        delYt[2] = delA[2] + P*delN[2] + delP[2]*N
        return delYt

    def del2Ytf(self, xyt, N, delN, del2N):
        """Trial function Laplacian"""
        del2A = self.del2Af(xyt)
        P = self.Pf(xyt)
        delP = self.delPf(xyt)
        del2P = self.del2Pf(xyt)
        del2Yt = [None, None, None]
        del2Yt[0] = del2A[0] + P*del2N[0] + 2*delP[0]*delN[0] + del2P[0]*N
        del2Yt[1] = del2A[1] + P*del2N[1] + 2*delP[1]*delN[1] + del2P[1]*N
        del2Yt[2] = del2A[2] + P*del2N[2] + 2*delP[2]*delN[2] + del2P[2]*N
        return del2Yt

    def Af(self, xyt):
        """Boundary condition function for 2D diffusion problems"""
        (x, y, t) = xyt
        c = self.cf(xyt)
        A = c[0]*x**2 + c[1]*x + c[2]*y**2 + c[3]*y + c[4]*t
        return A

    def delAf(self, xyt):
        (x, y, t) = xyt
        c = self.cf(xyt)
        delc = self.delcf(xyt)
        delA = [0, 0, 0]
        delA[0] = 2*c[0]*x + delc[0][0]*x**2 + c[1] + delc[1][0]*x \
                   + delc[2][0]*y**2 + delc[3][0]*y + delc[4][0]*t
        delA[1] = delc[0][1]*x**2 + delc[1][1]*x + 2*c[2]*y \
                   + delc[2][1]*y**2 + c[3] + delc[3][1]*y \
                   + delc[4][1]*t
        delA[2] = delc[0][2]*x**2 + delc[1][2]*x + delc[2][2]*y**2 \
                   + delc[3][2]*y + c[4] + delc[4][2]*t
        return delA

    def del2Af(self, xyt):
        (x, y, t) = xyt
        c = self.cf(xyt)
        delc = self.delcf(xyt)
        del2c = self.del2cf(xyt)
        del2A = [0, 0, 0]
        del2A[0] = 2*c[0] + 4*delc[0][0]*x + del2c[0][0]*x**2 \
                    + 2*delc[1][0] + del2c[1][0]*x \
                    + del2c[2][0]*y**2 + del2c[3][0]*y + del2c[4][0]*t
        del2A[1] = del2c[0][1]*x**2 + del2c[1][1]*x \
                    + 2*c[2] + 4*delc[2][1]*y + del2c[2][1]*y**2 \
                    + 2*delc[3][1] + del2c[3][1]*y + del2c[4][1]*t
        del2A[2] = del2c[0][2]*x**2 + del2c[1][2]*x + del2c[2][2]*y**2 \
                    + del2c[3][2]*y + 2*delc[4][2] + del2c[4][2]*t
        return del2A

    def cf(self, xyt):
        """Compute the coefficient vector for the boundary condition function"""
        (x, y, t) = xyt
        ((f0f, f1f), (g0f, g1f), (h0f, h1f)) = self.bcf
        f0 = f0f(xyt); f1 = f1f(xyt)
        g0 = g0f(xyt); g1 = g1f(xyt)
        h0 = h0f(xyt); h1 = h1f(xyt)
        c0 = ((1 - 2*x)*f0 + 2*x*f1 - g0 - h0) / (2*(1 - x)*x)
        c1 = ((2*x**2 - 1)*f0 - 2*x**2*f1 + g0 + h0) / (2*(1 - x)*x)
        c2 = (-f0 + (1 - 2*y)*g0 + 2*y*g1 - h0) / (2*(1 - y)*y)
        c3 = (f0 + (2*y**2 - 1)*g0 - 2*y**2*g1 + h0) / (2*(1 - y)*y)
        c4 = (f0 + g0 - h0) / (2*t)
        c = [c0, c1, c2, c3, c4]
        return c

    def delcf(self, xyt):
        """Compute the gradients of each coefficient."""
        (x, y, t) = xyt
        ((f0f, f1f), (g0f, g1f), (h0f, h1f)) = self.bcf
        (((df0_dxf, df0_dyf, df0_dtf), (df1_dxf, df1_dyf, df1_dtf)),
         ((dg0_dxf, dg0_dyf, dg0_dtf), (dg1_dxf, dg1_dyf, dg1_dtf)),
         ((dh0_dxf, dh0_dyf, dh0_dtf), (dh1_dxf, dh1_dyf, dh1_dtf))) = self.delbcf
        f0 = f0f(xyt); f1 = f1f(xyt)
        g0 = g0f(xyt); g1 = g1f(xyt)
        h0 = h0f(xyt); h1 = h1f(xyt)
        df0_dx = df0_dxf(xyt); df0_dy = df0_dyf(xyt); df0_dt = df0_dtf(xyt)
        df1_dx = df1_dxf(xyt); df1_dy = df1_dyf(xyt); df1_dt = df1_dtf(xyt)
        dg0_dx = dg0_dxf(xyt); dg0_dy = dg0_dyf(xyt); dg0_dt = dg0_dtf(xyt)
        dg1_dx = dg1_dxf(xyt); dg1_dy = dg1_dyf(xyt); dg1_dt = dg1_dtf(xyt)
        dh0_dx = dh0_dxf(xyt); dh0_dy = dh0_dyf(xyt); dh0_dt = dh0_dtf(xyt)
        dh1_dx = dh1_dxf(xyt); dh1_dy = dh1_dyf(xyt); dh1_dt = dh1_dtf(xyt)

        dc0_dx = ((-1 - 2*(-1 + x)*x)*f0 + g0 + h0
                   + x*(2*x*f1 - 2*g0 - 2*h0
                        + (-1 + x)*((-1 + 2*x)*df0_dx
                                    - 2*x*df1_dx + dg0_dx + dh0_dx))) / \
                 (2*(-1 + x)**2*x**2)
        dc0_dy = ((-1 + 2*x)*df0_dy - 2*x*df1_dy + dg0_dy + dh0_dy) / \
                 (2*(-1 + x)*x)
        dc0_dt = ((-1 + 2*x)*df0_dt - 2*x*df1_dt + dg0_dt + dh0_dy) / \
                 (2*(-1 + x)*x)

        dc1_dx = -((-1 - 2*(-1 + x)*x)*f0 + g0 + h0
                   + x*(2*x*f1 - 2*g0 - 2*h0
                        + (-1 + x)*((-1 + 2*x**2)*df0_dx - 2*x**2*df1_dx
                                     + dg0_dx + dh0_dx))) / \
                  (2*(-1 + x)**2*x**2)
        dc1_dy = -((-1 + 2*x**2)*df0_dy - 2*x**2*df1_dy + dg0_dy + dh0_dy) / \
                  (2*(-1 + x)*x)
        dc1_dt = -((-1 + 2*x**2)*df0_dt - 2*x**2*df1_dt + dg0_dt + dh0_dt) / \
                  (2*(-1 + x)*x)

        dc2_dx = (df0_dx + (-1 + 2*y)*dg0_dx - 2*y*dg1_dx + dh0_dx) / \
                 (2*(-1 + y)*y)
        dc2_dy = ((1 - 2*y)*f0 + (-1 - 2*(-1 + y)*y)*g0 + h0
                  + y*(2*y*g1 - 2*h0 +
                       (-1 + y)*(df0_dy + (-1 + 2*y)*dg0_dy
                                 - 2*y*dg1_dy + dh0_dy))) / \
                 (2*(-1 + y)**2*y**2)
        dc2_dt = (df0_dt + (-1 + 2*y)*dg0_dt - 2*y*dg1_dt + dh0_dt) / \
                 (2*(-1 + y)*y)

        dc3_dx = -(df0_dx + (-1 + 2*y**2)*dg0_dx - 2*y**2*dg1_dx + dh0_dx) / \
                 (2*(-1 + y)*y)
        dc3_dy = ((-1 + 2*y)*f0 + (1 + 2*(-1 + y)*y)*g0 - h0
                  + y*(-2*y*g1 + 2*h0
                       - (-1 + y)*(df0_dy + (-1 + 2*y**2)*dg0_dy
                                   - 2*y**2*dg1_dy + dh0_dy))) / \
                 (2*(-1 + y)**2*y**2)
        dc3_dt = -(df0_dt + (-1 + 2*y**2)*dg0_dt - 2*y**2*dg1_dt + dh0_dt) / \
                 (2*(-1 + y)*y)

        dc4_dx = (df0_dx + dg0_dx - dh0_dx) / (2*t)
        dc4_dy = (df0_dy + dg0_dy - dh0_dy) / (2*t)
        dc4_dt = (-f0 - g0 + h0 + t*(df0_dt + dg0_dt - dh0_dt)) / (2*t**2)
        delc = [
            [dc0_dx, dc0_dy, dc0_dt],
            [dc1_dx, dc1_dy, dc1_dt],
            [dc2_dx, dc2_dy, dc2_dt],
            [dc3_dx, dc3_dy, dc3_dt],
            [dc4_dx, dc4_dy, dc4_dt],
            ]
        return delc

    def del2cf(self, xyt):
        """Compute the Laplacians of each coefficient."""
        (x, y, t) = xyt
        ((f0f, f1f), (g0f, g1f), (h0f, h1f)) = self.bcf
        (((df0_dxf, df0_dyf, df0_dtf), (df1_dxf, df1_dyf, df1_dtf)),
        ((dg0_dxf, dg0_dyf, dg0_dtf), (dg1_dxf, dg1_dyf, dg1_dtf)),
        ((dh0_dxf, dh0_dyf, dh0_dtf), (dh1_dxf, dh1_dyf, dh1_dtf))) = self.delbcf
        (((d2f0_dx2f, d2f0_dy2f, d2f0_dt2f), (d2f1_dx2f, d2f1_dy2f, d2f1_dt2f)),
        ((d2g0_dx2f, d2g0_dy2f, d2g0_dt2f), (d2g1_dx2f, d2g1_dy2f, d2g1_dt2f)),
        ((d2h0_dx2f, d2h0_dy2f, d2h0_dt2f), (d2h1_dx2f, d2h1_dy2f, d2h1_dt2f))) = self.del2bcf

        f0 = f0f(xyt); f1 = f1f(xyt)
        g0 = g0f(xyt); g1 = g1f(xyt)
        h0 = h0f(xyt); h1 = h1f(xyt)

        df0_dx = df0_dxf(xyt); df0_dy = df0_dyf(xyt); df0_dt = df0_dtf(xyt)
        df1_dx = df1_dxf(xyt); df1_dy = df1_dyf(xyt); df1_dt = df1_dtf(xyt)
        dg0_dx = dg0_dxf(xyt); dg0_dy = dg0_dyf(xyt); dg0_dt = dg0_dtf(xyt)
        dg1_dx = dg1_dxf(xyt); dg1_dy = dg1_dyf(xyt); dg1_dt = dg1_dtf(xyt)
        dh0_dx = dh0_dxf(xyt); dh0_dy = dh0_dyf(xyt); dh0_dt = dh0_dtf(xyt)
        dh1_dx = dh1_dxf(xyt); dh1_dy = dh1_dyf(xyt); dh1_dt = dh1_dtf(xyt)

        d2f0_dx2 = d2f0_dx2f(xyt); d2f0_dy2 = d2f0_dy2f(xyt); d2f0_dt2 = d2f0_dt2f(xyt)
        d2f1_dx2 = d2f1_dx2f(xyt); d2f1_dy2 = d2f1_dy2f(xyt); d2f1_dt2 = d2f1_dt2f(xyt)
        d2g0_dx2 = d2g0_dx2f(xyt); d2g0_dy2 = d2g0_dy2f(xyt); d2g0_dt2 = d2g0_dt2f(xyt)
        d2g1_dx2 = d2g1_dx2f(xyt); d2g1_dy2 = d2g1_dy2f(xyt); d2g1_dt2 = d2g1_dt2f(xyt)
        d2h0_dx2 = d2h0_dx2f(xyt); d2h0_dy2 = d2h0_dy2f(xyt); d2h0_dt2 = d2h0_dt2f(xyt)
        d2h1_dx2 = d2h1_dx2f(xyt); d2h1_dy2 = d2h1_dy2f(xyt); d2h1_dt2 = d2h1_dt2f(xyt)

        d2c0_dx2 = (2*(-1 + 2*x)*(1 + (-1 + x)*x)*f0 - 4*x**3*f1 +
                    2*(g0 + h0) +
                    (-1 + x)*x*(6*g0 + 6*h0 + (-2 - 4*(-1 + x)*x)*df0_dx +
                                2*(dg0_dx + dh0_dx) +
                                x*(4*x*df1_dx - 4*dg0_dx - 4*dh0_dx +
                                (-1 + x)*((-1 + 2*x)*d2f0_dx2 - 2*x*d2f1_dx2 +
                                            d2g0_dx2 + d2h0_dx2)))) / \
                (2*(-1 + x)**3*x**3)
        d2c0_dy2 = ((-1 + 2*x)*d2f0_dy2 - 2*x*d2f1_dy2 + d2g0_dy2 + d2h0_dy2) / \
                (2*(-1 + x)*x)
        d2c0_dt2 = ((-1 + 2*x)*d2f0_dt2 - 2*x*d2f1_dt2 + d2g0_dt2 + d2h0_dt2) / \
                (2*(-1 + x)*x)

        d2c1_dx2 = ((2 - 2*x*(3 + x*(-3 + 2*x)))*f0 + 4*x**3*f1 - 2*(g0 + h0) +
                    (-1 + x)*x*(-6*g0 - 6*h0 + (2 + 4*(-1 + x)*x)*df0_dx -
                                2*(dg0_dx + dh0_dx) +
                                x*(-4*x*df1_dx + 4*dg0_dx + 4*dh0_dx -
                                (-1 + x) *((-1 + 2*x**2)*d2f0_dx2 -
                                            2*x*2*d2f1_dx2 + d2g0_dx2 +
                                            d2h0_dx2)))) / \
                (2*(-1 + x)**3*x**3)
        d2c1_dy2 = -((-1 + 2*x**2)*d2f0_dy2 - 2*x**2*d2f1_dy2 + d2g0_dy2 + d2h0_dy2) / \
                (2*(-1 + x)*x)
        d2c1_dt2 = -((-1 + 2*x**2)*d2f0_dt2 - 2*x**2*d2f1_dt2 + d2g0_dt2 + d2h0_dt2) / \
                (2*(-1 + x)*x)

        d2c2_dx2 = (d2f0_dx2 + (-1 + 2*y)*d2g0_dx2 - 2*y*d2g1_dx2 + d2h0_dx2) / \
                (2*(-1 + y)*y)
        d2c2_dy2 = ((-2 + 6*(-1 + y)*y)*f0 + 2*(-1 + 2*y)*(1 + (-1 + y)*y)*g0 -
                    4*y**3*g1 + 2*h0 +
                    (-1 + y)*y*(6*h0 + (2 - 4*y)*df0_dy +
                                (-2 - 4*(-1 + y)*y)*dg0_dy + 2*dh0_dy +
                                y*(4*y*dg1_dy - 4*dh0_dy +
                                    (-1 + y)*(d2f0_dy2 + (-1 + 2*y)*d2g0_dy2 -
                                            2*y*d2g1_dy2 + d2h0_dy2)))) / \
                (2*(-1 + y)**3*y**3)
        d2c2_dt2 = (d2f0_dt2 + (-1 + 2*y)*d2g0_dt2 - 2*y*d2g1_dt2 + d2h0_dt2) / \
                (2*(-1 + y)*y)

        d2c3_dx2 = -(d2f0_dx2 + (-1 + 2*y**2)*d2g0_dx2 -
                    2*y**2*d2g1_dx2 + d2h0_dx2) / \
                (2*(-1 + y)*y)
        d2c3_dy2 = ((-2 - 6*(-1 + y)*y)*f0 + (2 - 2*y*(3 + y*(-3 + 2*y)))*g0 +
                    4*y**3*g1 - 2*h0 +
                    (-1 + y)*y*(-6*h0 + (-2 + 4*y)*df0_dy +
                                (2 + 4*(-1 + y)*y)*dg0_dy - 2*dh0_dy +
                                y*(-4*y*dg1_dy + 4*dh0_dy -
                                (-1 + y)*(d2f0_dy2 + (-1 + 2*y**2)*d2g0_dy2 -
                                            2*y**2*d2g1_dy2 + d2h0_dy2)))) / \
                (2*(-1 + y)**3*y**3)
        d2c3_dt2 = -(d2f0_dt2 + (-1 + 2*y**2)*d2g0_dt2 -
                    2*y**2*d2g1_dt2 + d2h0_dt2) / \
                (2*(-1 + y)*y)

        d2c4_dx2 = (d2f0_dx2 + d2g0_dx2 - d2h0_dx2) / (2*t)
        d2c4_dy2 = (d2f0_dy2f(xyt) + d2g0_dy2f(xyt) - d2h0_dy2f(xyt)) / \
                (2*t)
        d2c4_dt2 = (2*f0 + 2*g0 - 2*h0 +
                    t*(-2*df0_dt - 2*dg0_dt + 2*dh0_dt +
                    t*(d2f0_dt2 + d2g0_dt2 - d2h0_dt2))) / \
                (2*t**3)
        del2c = [
            [d2c0_dx2, d2c0_dy2, d2c0_dt2],
            [d2c1_dx2, d2c1_dy2, d2c1_dt2],
            [d2c2_dx2, d2c2_dy2, d2c2_dt2],
            [d2c3_dx2, d2c3_dy2, d2c3_dt2],
            [d2c4_dx2, d2c4_dy2, d2c4_dt2],
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
        delP = [0, 0, 0]
        delP[0] = (1 - 2*x)*y*(1 - y)*t
        delP[1] = x*(1 - x)*(1 - 2*y)*t
        delP[2] = x*(1 - x)*y*(1 - y)
        return delP

    def del2Pf(self, xyt):
        """Network coefficient function Laplacian"""
        (x, y, t) = xyt
        del2P = [0, 0, 0]
        del2P[0] = -2*y*(1 - y)*t
        del2P[1] = -2*x*(1 - x)*t
        del2P[2] = 0
        return del2P

#################


# Self-test code

from math import pi, sin, cos
import numpy as np

if __name__ == '__main__':

    # Test BC - 0 everywhere, for all time
    f0f = lambda xyt: 0
    f1f = lambda xyt: 0
    g0f = lambda xyt: 0
    g1f = lambda xyt: 0
    h0f = lambda xyt: sin(pi*xyt[0])*sin(pi*xyt[1])
    h1f = lambda xyt: None
    bcf = [[f0f, f1f], [g0f, g1f], [h0f, h1f]]

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
    dh0_dxf = lambda xyt: pi*cos(pi*xyt[0])*sin(pi*xyt[1])
    dh0_dyf = lambda xyt: pi*sin(pi*xyt[0])*cos(pi*xyt[1])
    dh0_dtf = lambda xyt: 0
    dh1_dxf = lambda xyt: None
    dh1_dyf = lambda xyt: None
    dh1_dtf = lambda xyt: None
    delbcf = [[[df0_dxf, df0_dyf, df0_dtf], [df1_dxf, df1_dyf, df1_dtf]],
              [[dg0_dxf, dg0_dyf, dg0_dtf], [dg1_dxf, dg1_dyf, dg1_dtf]],
              [[dh0_dxf, dh0_dyf, dh0_dtf], [dh1_dxf, dh1_dyf, dh1_dtf]]]

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
    d2h0_dx2f = lambda xyt: -pi**2*sin(pi*xyt[0])*sin(pi*xyt[1])
    d2h0_dy2f = lambda xyt: -pi**2*sin(pi*xyt[0])*sin(pi*xyt[1])
    d2h0_dt2f = lambda xyt: 0
    d2h1_dx2f = lambda xyt: None
    d2h1_dy2f = lambda xyt: None
    d2h1_dt2f = lambda xyt: None
    del2bcf = [[[d2f0_dx2f, d2f0_dy2f, d2f0_dt2f], [d2f1_dx2f, d2f1_dy2f, d2f1_dt2f]],
               [[d2g0_dx2f, d2g0_dy2f, d2g0_dt2f], [d2g1_dx2f, d2g1_dy2f, d2g1_dt2f]],
               [[d2h0_dx2f, d2h0_dy2f, d2h0_dt2f], [d2h1_dx2f, d2h1_dy2f, d2h1_dt2f]]]

    # Reference values for tests.
    bc_ref = [[0, 0],
              [0, 0],
              [0.9510565162951535, None]]
    delbc_ref = [[[0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0]],
                  [[0.9708055193627333, 1.82952e-16, 0], [None, None, None]]]
    del2bc_ref = [[[0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0]],
                  [[-9.38655, -9.38655, 0], [None, None, None]]]
    c_ref = [-1.9813677422815699, 1.9813677422815699, -1.902113032590307,
             1.902113032590307, -0.7925470969126279]
    delc_ref = [[-0.371372, -3.8115e-16, 0],
                 [0.371372, 3.8115e-16, 0],
                 [-1.94161, 0, 0],
                 [1.94161, 0, 0],
                 [-0.809005, -1.5246e-16, 1.32091]]
    del2c_ref = [[3.6628707927817716, 19.55531578939867, -0.0],
                 [-3.6628707927817716, -19.55531578939867, -0.0],
                 [18.77310315782272, 3.556198897100252, -0.0],
                 [-18.77310315782272, -3.556198897100252, 0.0],
                 [7.822126315759467, 7.822126315759467, -4.403039427292378]]
    A_ref = 0.475528
    delA_ref = [0.48540275968136676, -9.147597742350633e-17, -1.1102230246251565e-16]
    del2A_ref = [-4.693275789455675, -4.6932757894556785, -4.440892098500626e-16]
    P_ref = 0.036
    delP_ref = [0.03, 0.0, 0.06]
    del2P_ref = [-0.3, -0.288, 0]
    Yt_ref = 0.4799562581475769
    delYt_ref = [0.4926927596813667, 0.007199999999999909, 0.018179999999999887]
    del2Yt_ref = [-4.720215789455675, -4.720779789455678, 0.047879999999999555]

    # Additional test variables.
    N_ref = 0.123
    delN_ref = [0.1, 0.2, 0.3]
    del2N_ref = [0.11, 0.22, 0.33]

    # Test all functions near the center of the domain.
    xyt = [0.4, 0.5, 0.6]
    print("xyt =", xyt)

    # Create a new trial function object.
    tf = Diff2DTrialFunction(bcf, delbcf, del2bcf)

    print("Testing BC functions.")
    for i in range(len(tf.bcf)):
        for (j, f) in enumerate(tf.bcf[i]):
            bc = f(xyt)
            if ((bc_ref[i][j] is not None and not np.isclose(bc, bc_ref[i][j]))
                or (bc_ref[i][j] is None and bc is not None)):
                print("ERROR: bc[%d][%d] = %s, vs ref %s" % (i, j, bc, bc_ref[i][j]))

    print("Testing BC gradient functions.")
    for i in range(len(tf.delbcf)):
        for j in range(len(tf.delbcf[i])):
            for (k, f) in enumerate(tf.delbcf[i][j]):
                delbc = f(xyt)
                if ((delbc_ref[i][j][k] is not None and not np.isclose(delbc, delbc_ref[i][j][k]))
                    or (delbc_ref[i][j][k] is None and delbc is not None)):
                    print("ERROR: delbc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, delbc, delbc_ref[i][j][k]))

    print("Testing BC Laplacian functions.")
    for i in range(len(tf.del2bcf)):
        for j in range(len(tf.del2bcf[i])):
            for (k, f) in enumerate(tf.del2bcf[i][j]):
                del2bc = f(xyt)
                if ((del2bc_ref[i][j][k] is not None and not np.isclose(del2bc, del2bc_ref[i][j][k]))
                    or (del2bc_ref[i][j][k] is None and del2bc is not None)):
                    print("ERROR: del2bc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, del2bc, del2bc_ref[i][j][k]))

    print("Testing coefficient functions.")
    c = tf.cf(xyt)
    for (i, cc) in enumerate(c):
        if not np.isclose(cc, c_ref[i]):
            print("ERROR: c[%d] = %s, vs ref %s" % (i, cc, c_ref[i]))

    print("Testing coefficient gradient functions.")
    delc = tf.delcf(xyt)
    for i in range(len(c)):
        for (j, delci) in enumerate(delc[i]):
            if not np.isclose(delci, delc_ref[i][j]):
                print("ERROR: delc[%d][%d] = %s, vs ref %s" % (i, j, delci, delc_ref[i][j]))

    print("Testing coefficient Laplacian functions.")
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
            print("ERROR: delA[%d] = %s, vs ref %s" % (delAi, delA_ref[i]))

    print("Testing boundary condition function Laplacian.")
    del2A = tf.del2Af(xyt)
    for (i, del2Ai) in enumerate(del2A):
        if not np.isclose(del2Ai, del2A_ref[i]):
            print("ERROR: del2A[%d] = %s, vs ref %s" % (del2Ai, del2A_ref[i]))

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
            print("ERROR: delYt[%d] = %s, vs ref %s" % (delYti, delYt_ref[i]))

    print("Testing trial function Laplacian.")
    del2Yt = tf.del2Ytf(xyt, N_ref, delN_ref, del2N_ref)
    for (i, del2Yti) in enumerate(del2Yt):
        if not np.isclose(del2Yti, del2Yt_ref[i]):
            print("ERROR: del2Yt[%d] = %s, vs ref %s" % (del2Yti, del2Yt_ref[i]))
