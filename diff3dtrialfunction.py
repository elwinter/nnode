"""
Diff3DTrialFunction - Class implementing the trial function for 3-D diffusion
problems

The trial function takes the form:

Yt(x,y,z,t) = A(x,y,z,t) + P(x,y,z,t)N(x,y,z,t,p)

where:

A(x,y,z,t) = boundary condition function that reduces to BC at boundaries
P(x,y,z,t) = network coefficient function that vanishes at boundaries
N(x,y,z,t,p) = scalar output of neural network with parameter vector p

Example:
    Create a default Diff3DTrialFunction object
        Yt = Diff3DTrialFunction()

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


class Diff3DTrialFunction():
    """Trial function for 3D diffusion problems."""


    # Public methods

    def __init__(self, bcf, delbcf, del2bcf):
        """Constructor"""
        self.bcf = bcf
        self.delbcf = delbcf
        self.del2bcf = del2bcf

    def Ytf(self, xyzt, N):
        """Trial function"""
        A = self.Af(xyzt)
        P = self.Pf(xyzt)
        Yt = A + P*N
        return Yt

    def delYtf(self, xyzt, N, delN):
        """Trial function gradient"""
        delA = self.delAf(xyzt)
        P = self.Pf(xyzt)
        delP = self.delPf(xyzt)
        delYt = [None, None, None, None]
        delYt[0] = delA[0] + P*delN[0] + delP[0]*N
        delYt[1] = delA[1] + P*delN[1] + delP[1]*N
        delYt[2] = delA[2] + P*delN[2] + delP[2]*N
        delYt[3] = delA[3] + P*delN[3] + delP[3]*N
        return delYt

    def del2Ytf(self, xyzt, N, delN, del2N):
        """Trial function Laplacian"""
        del2A = self.del2Af(xyzt)
        P = self.Pf(xyzt)
        delP = self.delPf(xyzt)
        del2P = self.del2Pf(xyzt)
        del2Yt = [None, None, None, None]
        del2Yt[0] = del2A[0] + P*del2N[0] + 2*delP[0]*delN[0] + del2P[0]*N
        del2Yt[1] = del2A[1] + P*del2N[1] + 2*delP[1]*delN[1] + del2P[1]*N
        del2Yt[2] = del2A[2] + P*del2N[2] + 2*delP[2]*delN[2] + del2P[2]*N
        del2Yt[3] = del2A[3] + P*del2N[3] + 2*delP[3]*delN[3] + del2P[3]*N
        return del2Yt

    def Af(self, xyzt):
        """Boundary condition function"""
        (x, y, z, t) = xyzt
        (cx1, cx2, cy1, cy2, cz1, cz2, ct1) = self.cf(xyzt)
        A = cx1*x + cx2*x**2 + cy1*y + cy2*y**2 + cz1*z + cz2*z**2 + ct1*t
        return A

    def delAf(self, xyzt):
        """Gradient of boundary condition function"""
        (x, y, z, t) = xyzt
        (cx1, cx2, cy1, cy2, cz1, cz2, ct1) = self.cf(xyzt)
        ((dcx1_dx, dcx1_dy, dcx1_dz, dcx1_dt),
         (dcx2_dx, dcx2_dy, dcx2_dz, dcx2_dt),
         (dcy1_dx, dcy1_dy, dcy1_dz, dcy1_dt),
         (dcy2_dx, dcy2_dy, dcy2_dz, dcy2_dt),
         (dcz1_dx, dcz1_dy, dcz1_dz, dcz1_dt),
         (dcz2_dx, dcz2_dy, dcz2_dz, dcz2_dt),
         (dct1_dx, dct1_dy, dct1_dz, dct1_dt),
        ) = self.delcf(xyzt)
        dA_dx = (cx1 + dcx1_dx*x) + (cx2*2*x + dcx2_dx*x**2) + (dcy1_dx*y) + (dcy2_dx*y**2) + (dcz1_dx*z) + (dcz2_dx*z**2) + (dct1_dx*t)
        dA_dy = (dcx1_dy*x) + (dcx2_dy*x**2) + (cy1 + dcy1_dy*y) + (cy2*2*y + dcy2_dy*y**2) + (dcz1_dy*z) + (dcz2_dy*z**2) + (dct1_dy*t)
        dA_dz = (dcx1_dz*x) + (dcx2_dz*x**2) + (dcy1_dz*y) + (dcy2_dz*y**2) + (cz1 + dcz1_dz*z) + (cz2*2*z + dcz2_dz*z**2) + (dct1_dz*t)
        dA_dt = (dcx1_dt*x) + (dcx2_dt*x**2) + (dcy1_dt*y) + (dcy2_dt*y**2) + (dcz1_dt*z) + (dcz2_dt*z**2) + (ct1 + dct1_dt*t)
        delA = [dA_dx, dA_dy, dA_dz, dA_dt]
        return delA

    def del2Af(self, xyzt):
        (x, y, z, t) = xyzt
        (cx1, cx2, cy1, cy2, cz1, cz2, ct1) = self.cf(xyzt)
        ((dcx1_dx, dcx1_dy, dcx1_dz, dcx1_dt),
         (dcx2_dx, dcx2_dy, dcx2_dz, dcx2_dt),
         (dcy1_dx, dcy1_dy, dcy1_dz, dcy1_dt),
         (dcy2_dx, dcy2_dy, dcy2_dz, dcy2_dt),
         (dcz1_dx, dcz1_dy, dcz1_dz, dcz1_dt),
         (dcz2_dx, dcz2_dy, dcz2_dz, dcz2_dt),
         (dct1_dx, dct1_dy, dct1_dz, dct1_dt),
        ) = self.delcf(xyzt)
        ((d2cx1_dx2, d2cx1_dy2, d2cx1_dz2, d2cx1_dt2),
         (d2cx2_dx2, d2cx2_dy2, d2cx2_dz2, d2cx2_dt2),
         (d2cy1_dx2, d2cy1_dy2, d2cy1_dz2, d2cy1_dt2),
         (d2cy2_dx2, d2cy2_dy2, d2cy2_dz2, d2cy2_dt2),
         (d2cz1_dx2, d2cz1_dy2, d2cz1_dz2, d2cz1_dt2),
         (d2cz2_dx2, d2cz2_dy2, d2cz2_dz2, d2cz2_dt2),
         (d2ct1_dx2, d2ct1_dy2, d2ct1_dz2, d2ct1_dt2),
        ) = self.del2cf(xyzt)

        d2A_dx2 = (2*dcx1_dx + d2cx1_dx2*x) + (cx2*2 + 2*dcx2_dx*2*x + d2cx2_dx2*x**2) + (d2cy1_dx2*y) + (d2cy2_dx2*y**2) + (d2cz1_dx2*z) + (d2cz2_dx2*z**2) + (d2ct1_dx2*t)
        d2A_dy2 = (d2cx1_dy2*x) + (d2cx2_dy2*x**2) + (2*dcy1_dy + d2cy1_dy2*y) + (cy2*2 + 2*dcy2_dy*2*y + d2cy2_dy2*y**2) + (d2cz1_dy2*z) + (d2cz2_dy2*z**2) + (d2ct1_dy2*t)
        d2A_dz2 = (d2cx1_dz2*x) + (d2cx2_dz2*x**2) + (d2cy1_dz2*y) + (d2cy2_dz2*y**2) + (2*dcz1_dz + d2cz1_dz2*z) + (cz2*2 + 2*dcz2_dz*2*z + d2cz2_dz2*z**2) + (d2ct1_dz2*t)
        d2A_dt2 = (d2cx1_dt2*x) + (d2cx2_dt2*x**2) + (d2cy1_dt2*y) + (d2cy2_dt2*y**2) + (d2cz1_dt2*z) + (d2cz2_dt2*z**2) + (2*dct1_dt + d2ct1_dt2*t)
        del2A = [d2A_dx2, d2A_dy2, d2A_dz2, d2A_dt2]
        return del2A

    def cf(self, xyzt):
        """Compute the coefficient vector for the boundary condition function"""
        (x, y, z, t) = xyzt
        ((f0f, f1f), (g0f, g1f), (h0f, h1f), (Y0f, Y1f)) = self.bcf
        f0 = f0f(xyzt); f1 = f1f(xyzt)
        g0 = g0f(xyzt); g1 = g1f(xyzt)
        h0 = h0f(xyzt); h1 = h1f(xyzt)
        Y0 = Y0f(xyzt); Y1 = Y1f(xyzt)
        cx1 = (f0*(3*x**2 - 2) - 3*f1*x**2 + g0 + h0 + Y0)/(3*x*(1 - x))
        cx2 = (f0*(2 - 3*x) + 3*f1*x - g0 - h0 - Y0)/(3*x*(1 - x))
        cy1 = (f0 + g0*(3*y**2 - 2) - 3*g1*y**2 + h0 + Y0)/(3*y*(1 - y))
        cy2 = (-f0 + g0*(2 - 3*y) + 3*g1*y - h0 - Y0)/(3*y*(1 - y))
        cz1 = (f0 + g0 + h0*(3*z**2 - 2) - 3*h1**z**2 + Y0)/(3*z*(1 - z))
        cz2 = (-f0 - g0 + h0*(2 - 3*z) + 3*h1*z - Y0)/(3*z*(1 - z))
        ct1 = (f0 + g0 + h0 - 2*Y0)/(3*t)
        c = [cx1, cx2, cy1, cy2, cz1, cz2, ct1]
        return c

    def delcf(self, xyzt):
        """Compute the gradients of each coefficient."""
        (x, y, z, t) = xyzt
        ((f0f, f1f), (g0f, g1f), (h0f, h1f), (Y0f, Y1f)) = self.bcf
        (((df0_dxf, df0_dyf, df0_dzf, df0_dtf), (df1_dxf, df1_dyf, df1_dzf, df1_dtf)),
         ((dg0_dxf, dg0_dyf, dg0_dzf, dg0_dtf), (dg1_dxf, dg1_dyf, dg1_dzf, dg1_dtf)),
         ((dh0_dxf, dh0_dyf, dh0_dzf, dh0_dtf), (dh1_dxf, dh1_dyf, dh1_dzf, dh1_dtf)),
         ((dY0_dxf, dY0_dyf, dY0_dzf, dY0_dtf), (dY1_dxf, dY1_dyf, dY1_dzf, dY1_dtf))) = self.delbcf
        f0 = f0f(xyzt); f1 = f1f(xyzt)
        g0 = g0f(xyzt); g1 = g1f(xyzt)
        h0 = h0f(xyzt); h1 = h1f(xyzt)
        Y0 = Y0f(xyzt); Y1 = Y1f(xyzt)
        df0_dx = df0_dxf(xyzt); df0_dy = df0_dyf(xyzt); df0_dz = df0_dzf(xyzt); df0_dt = df0_dtf(xyzt)
        df1_dx = df1_dxf(xyzt); df1_dy = df1_dyf(xyzt); df1_dz = df1_dzf(xyzt); df1_dt = df1_dtf(xyzt)
        dg0_dx = dg0_dxf(xyzt); dg0_dy = dg0_dyf(xyzt); dg0_dz = dg0_dzf(xyzt); dg0_dt = dg0_dtf(xyzt)
        dg1_dx = dg1_dxf(xyzt); dg1_dy = dg1_dyf(xyzt); dg1_dz = dg1_dzf(xyzt); dg1_dt = dg1_dtf(xyzt)
        dh0_dx = dh0_dxf(xyzt); dh0_dy = dh0_dyf(xyzt); dh0_dz = dh0_dzf(xyzt); dh0_dt = dh0_dtf(xyzt)
        dh1_dx = dh1_dxf(xyzt); dh1_dy = dh1_dyf(xyzt); dh1_dz = dh1_dzf(xyzt); dh1_dt = dh1_dtf(xyzt)
        dY0_dx = dY0_dxf(xyzt); dY0_dy = dY0_dyf(xyzt); dY0_dz = dY0_dzf(xyzt); dY0_dt = dh0_dtf(xyzt)
        dY1_dx = dY1_dxf(xyzt); dY1_dy = dY1_dyf(xyzt); dY1_dz = dY1_dzf(xyzt); dY1_dt = dh1_dtf(xyzt)

        dcx1_dx = -((-2 + (4 - 3*x)*x)*f0 + g0 + h0 + Y0 +
                    x*(3*x*f1 - 2*g0 - 2*h0 - 2*Y0 +
                       (-1 + x)*((-2 + 3*x**2)*df0_dx - 3*x**2*df1_dx +
                                 dg0_dx + dh0_dx + dY0_dx)))/ \
                  (3*(-1 + x)**2*x**2)
        dcx1_dy = -((-2 + 3*x**2)*df0_dy - 3*x**2*df1_dy + dg0_dy + dh0_dy +
                    dY0_dy)/ \
                  (3*(-1 + x)*x)
        dcx1_dz = -((-2 + 3*x**2)*df0_dz - 3*x**2*df1_dz + dg0_dz + dh0_dz +
                    dY0_dz)/ \
                  (3*(-1 + x)*x)
        dcx1_dt = -((-2 + 3*x**2)*df0_dt - 3*x**2*df1_dt + dg0_dt + dh0_dt +
                    dY0_dt)/ \
                  (3*(-1 + x)*x)

        dcx2_dx = ((-2 + (4 - 3*x)*x)*f0 + g0 + h0 + Y0 +
                   x*(3*x*f1 - 2*g0 - 2*h0 - 2*Y0 +
                      (-1 + x)*((-2 + 3*x)*df0_dx - 3*x*df1_dx + dg0_dx +
                                dh0_dx + dY0_dx)))/ \
                  (3*(-1 + x)**2*x**2)
        dcx2_dy = ((-2 + 3*x)*df0_dy - 3*x*df1_dy + dg0_dy + dh0_dy + dY0_dy)/ \
                  (3*(-1 + x)*x)
        dcx2_dz = ((-2 + 3*x)*df0_dz - 3*x*df1_dz + dg0_dz + dh0_dz + dY0_dz)/ \
                  (3*(-1 + x)*x)
        dcx2_dt = ((-2 + 3*x)*df0_dt - 3*x*df1_dt + dg0_dt + dh0_dt + dY0_dt)/ \
                  (3*(-1 + x)*x)

        dcy1_dx = -(df0_dx + (-2 + 3*y**2)*dg0_dx - 3*y**2*dg1_dx + dh0_dx + dY0_dx)/ \
                  (3*(-1 + y)*y)
        dcy1_dy = ((-1 + 2*y)*f0 + (2 + y*(-4 + 3*y))*g0 - h0 - Y0 +
                   y*(-3*y*g1 + 2*h0 + 2*Y0 - (-1 + y)*
                      (df0_dy + (-2 + 3*y**2)*dg0_dy - 3*y**2*dg1_dy + dh0_dy + dY0_dy)))/ \
                  (3*(-1 + y)**2*y**2)
        dcy1_dz = -(df0_dz + (-2 + 3*y**2)*dg0_dz - 3*y**2*dg1_dz + dh0_dz + dY0_dz)/ \
                  (3*(-1 + y)*y)
        dcy1_dt = -(df0_dt + (-2 + 3*y**2)*dg0_dt - 3*y**2*dg1_dt + dh0_dt + dY0_dt)/ \
                  (3*(-1 + y)*y)

        dcy2_dx = (df0_dx + (-2 * 3*y)*dg0_dx - 3*y*dg1_dx + dh0_dx + dY0_dx)/ \
                  (3*(-1 + y)*y)
        dcy2_dy = ((1 - 2*y)*f0 + (-2 + (4 - 3*y)*y)*g0 + h0 + Y0 +
                   y*(3*y*g1 - 2*h0 - 2*Y0 + (-1 + y)*
                      (df0_dy + (-2 + 3*y)*dg0_dy - 3*y*dg1_dy + dh0_dy + dY0_dy)))/ \
                  (3*(-1 + y)**2*y**2)
        dcy2_dz = (df0_dz + (-2 + 3*y)*dg0_dz - 3*y*dg1_dz + dh0_dz + dY0_dz)/ \
                  (3*(-1 + y)*y)
        dcy2_dt = (df0_dt + (-2 + 3*y)*dg0_dt - 3*y*dg1_dt + dh0_dt + dY0_dt)/ \
                  (3*(-1 + y)*y)

        dcz1_dx = -(df0_dx + 2*h0*dg0_dx + (3*z**2 + 2*g0)*dh0_dx -
                    3*z**2*dh1_dx + dY0_dx)/ \
                  (3*(-1 + z)*z)
        dcz1_dy = -(df0_dy + 2*h0*dg0_dy + (3*z**2 + 2*g0)*dh0_dy -
                    3*z**2*dh1_dy + dY0_dy)/ \
                  (3*(-1 + z)*z)
        dcz1_dz = ((-1 + 2*z)*f0 - Y0 +
                   h0*((-2 + 4*z)*g0 + z*(3*z - 2*(-1 + z)*dg0_dz)) +
                   z*(-3*z*h1 + 2*Y0 -
                      (-1 + z)*(df0_dz + (3*z**2 + 2*g0)*dh0_dz -
                                 3*z**2*dh1_dz + dY0_dz)))/ \
                  (3*(-1 + z)**2*z**2)
        dcz1_dt = -(df0_dt + 2*h0*dg0_dt + (3*z**2 + 2*g0)*dh0_dt -
                    3*z**2*dh1_dt + dY0_dt)/ \
                  (3*(-1 + z)*z)

        dcz2_dx = (df0_dx + dg0_dx + (-2 + 3*z)*dh0_dx - 3*z*dh1_dx + dY0_dx)/ \
                  (3*(-1 + z)*z)
        dcz2_dy = (df0_dy + dg0_dy + (-2 + 3*z)*dh0_dy - 3*z*dh1_dy + dY0_dy)/ \
                  (3*(-1 + z)*z)
        dcz2_dz = ((1 - 2*z)*f0 + (1 - 2*z)*g0 - 2*h0 + Y0 +
                   z*((4 - 3*z)*h0 + 3*z*h1 - 2*Y0 +
                      (-1 + z)*(df0_dz + dg0_dz + (-2 + 3*z)*dh0_dz -
                      -3*z*dh1_dz + dY0_dz)))/ \
                  (3*(-1 + z)**2*z**2)
        dcz2_dt = (df0_dt + dg0_dt + (-2 + 3*z)*dh0_dt - 3*z*dh1_dt + dY0_dt)/ \
                  (3*(-1 + z)*z)

        dct1_dx = (df0_dx + dg0_dx + dh0_dx - 2*dY0_dx)/(3*t)
        dct1_dy = (df0_dy + dg0_dy + dh0_dy - 2*dY0_dy)/(3*t)
        dct1_dz = (df0_dz + dg0_dz + dh0_dz - 2*dY0_dz)/(3*t)
        dct1_dt = -(f0 + g0 + h0 - 2*Y0 -
                    t*(df0_dt + dg0_dt + dh0_dt - 2*dY0_dt))/ \
                  (3*t**2)

        delc = [
            [dcx1_dx, dcx1_dy, dcx1_dz, dcx1_dt],
            [dcx2_dx, dcx2_dy, dcx2_dz, dcx2_dt],
            [dcy1_dx, dcy1_dy, dcy1_dz, dcy1_dt],
            [dcy2_dx, dcy2_dy, dcy2_dz, dcy2_dt],
            [dcz1_dx, dcz1_dy, dcz1_dz, dcz1_dt],
            [dcz2_dx, dcz2_dy, dcz2_dz, dcz2_dt],
            [dct1_dx, dct1_dy, dct1_dz, dct1_dt],
            ]
        return delc

    def del2cf(self, xyzt):
        """Compute the Laplacians of each coefficient."""
        (x, y, z, t) = xyzt
        ((f0f, f1f), (g0f, g1f), (h0f, h1f), (Y0f, Y1f)) = self.bcf
        (((df0_dxf, df0_dyf, df0_dzf, df0_dtf), (df1_dxf, df1_dyf, df1_dzf, df1_dtf)),
         ((dg0_dxf, dg0_dyf, dg0_dzf, dg0_dtf), (dg1_dxf, dg1_dyf, dg1_dzf, dg1_dtf)),
         ((dh0_dxf, dh0_dyf, dh0_dzf, dh0_dtf), (dh1_dxf, dh1_dyf, dh1_dzf, dh1_dtf)),
         ((dY0_dxf, dY0_dyf, dY0_dzf, dY0_dtf), (dY1_dxf, dY1_dyf, dY1_dzf, dY1_dtf))) = self.delbcf
        (((d2f0_dx2f, d2f0_dy2f, d2f0_dz2f, d2f0_dt2f), (d2f1_dx2f, d2f1_dy2f, d2f1_dz2f, d2f1_dt2f)),
         ((d2g0_dx2f, d2g0_dy2f, d2g0_dz2f, d2g0_dt2f), (d2g1_dx2f, d2g1_dy2f, d2g1_dz2f, d2g1_dt2f)),
         ((d2h0_dx2f, d2h0_dy2f, d2h0_dz2f, d2h0_dt2f), (d2h1_dx2f, d2h1_dy2f, d2h1_dz2f, d2h1_dt2f)),
         ((d2Y0_dx2f, d2Y0_dy2f, d2Y0_dz2f, d2Y0_dt2f), (d2Y1_dx2f, d2Y1_dy2f, d2Y1_dz2f, d2Y1_dt2f))) = self.del2bcf
        f0 = f0f(xyzt); f1 = f1f(xyzt)
        g0 = g0f(xyzt); g1 = g1f(xyzt)
        h0 = h0f(xyzt); h1 = h1f(xyzt)
        Y0 = Y0f(xyzt); Y1 = Y1f(xyzt)
        df0_dx = df0_dxf(xyzt); df0_dy = df0_dyf(xyzt); df0_dz = df0_dzf(xyzt); df0_dt = df0_dtf(xyzt)
        df1_dx = df1_dxf(xyzt); df1_dy = df1_dyf(xyzt); df1_dz = df1_dzf(xyzt); df1_dt = df1_dtf(xyzt)
        dg0_dx = dg0_dxf(xyzt); dg0_dy = dg0_dyf(xyzt); dg0_dz = dg0_dzf(xyzt); dg0_dt = dg0_dtf(xyzt)
        dg1_dx = dg1_dxf(xyzt); dg1_dy = dg1_dyf(xyzt); dg1_dz = dg1_dzf(xyzt); dg1_dt = dg1_dtf(xyzt)
        dh0_dx = dh0_dxf(xyzt); dh0_dy = dh0_dyf(xyzt); dh0_dz = dh0_dzf(xyzt); dh0_dt = dh0_dtf(xyzt)
        dh1_dx = dh1_dxf(xyzt); dh1_dy = dh1_dyf(xyzt); dh1_dz = dh1_dzf(xyzt); dh1_dt = dh1_dtf(xyzt)
        dY0_dx = dY0_dxf(xyzt); dY0_dy = dY0_dyf(xyzt); dY0_dz = dY0_dzf(xyzt); dY0_dt = dY0_dtf(xyzt)
        dY1_dx = dY1_dxf(xyzt); dY1_dy = dY1_dyf(xyzt); dY1_dz = dY1_dzf(xyzt); dY1_dt = dY1_dtf(xyzt)
        d2f0_dx2 = d2f0_dx2f(xyzt); d2f0_dy2 = d2f0_dy2f(xyzt); d2f0_dz2 = d2f0_dz2f(xyzt); d2f0_dt2 = d2f0_dt2f(xyzt)
        d2f1_dx2 = d2f1_dx2f(xyzt); d2f1_dy2 = d2f1_dy2f(xyzt); d2f1_dz2 = d2f1_dz2f(xyzt); d2f1_dt2 = d2f1_dt2f(xyzt)
        d2g0_dx2 = d2g0_dx2f(xyzt); d2g0_dy2 = d2g0_dy2f(xyzt); d2g0_dz2 = d2g0_dz2f(xyzt); d2g0_dt2 = d2g0_dt2f(xyzt)
        d2g1_dx2 = d2g1_dx2f(xyzt); d2g1_dy2 = d2g1_dy2f(xyzt); d2g1_dz2 = d2g1_dz2f(xyzt); d2g1_dt2 = d2g1_dt2f(xyzt)
        d2h0_dx2 = d2h0_dx2f(xyzt); d2h0_dy2 = d2h0_dy2f(xyzt); d2h0_dz2 = d2h0_dz2f(xyzt); d2h0_dt2 = d2h0_dt2f(xyzt)
        d2h1_dx2 = d2h1_dx2f(xyzt); d2h1_dy2 = d2h1_dy2f(xyzt); d2h1_dz2 = d2h1_dz2f(xyzt); d2h1_dt2 = d2h1_dt2f(xyzt)
        d2Y0_dx2 = d2Y0_dx2f(xyzt); d2Y0_dy2 = d2Y0_dy2f(xyzt); d2Y0_dz2 = d2Y0_dz2f(xyzt); d2Y0_dt2 = d2Y0_dt2f(xyzt)
        d2Y1_dx2 = d2Y1_dx2f(xyzt); d2Y1_dy2 = d2Y1_dy2f(xyzt); d2Y1_dz2 = d2Y1_dz2f(xyzt); d2Y1_dt2 = d2Y1_dt2f(xyzt)

        d2cx1_dx2 = (2*(2 - 3*x*(2 + (-2 + x)*x))*f0 + 6*x**3*f1 -
                     2*(g0 + h0 + Y0) - (-1 + x)*x*
                     (6*g0 + 6*h0 + 6*Y0 + (-4 + 8*x - 6*x**2)*df0_dx +
                      2*(dg0_dx + dh0_dx + dY0_dx) +
                      x*(6*x*df1_dx - 4*dg0_dx - 4*dh0_dx - 4*dY0_dx +
                         (-1 + x)*((-2 + 3*x**2)*d2f0_dx2 - 3*x**2*d2f1_dx2 +
                         d2g0_dx2 + d2h0_dx2 + d2Y0_dx2))))/ \
                    (3*(-1 + x)**3*x**3)
        d2cx1_dy2 = -((-2 + 3*x**2)*d2f0_dy2 - 3*x**2*d2f1_dy2 + d2g0_dy2 +
                      d2h0_dy2 + d2Y0_dy2)/ \
                    (3*(-1 + x)*x)
        d2cx1_dz2 = -((-2 + 3*x**2)*d2f0_dz2 - 3*x**2*d2f1_dz2 + d2g0_dz2 +
                      d2h0_dz2 + d2Y0_dz2)/ \
                    (3*(-1 + x)*x)
        d2cx1_dt2 = -((-2 + 3*x**2)*d2f0_dt2 - 3*x**2*d2f1_dt2 + d2g0_dt2 +
                      d2h0_dt2 + d2Y0_dt2)/ \
                    (3*(-1 + x)*x)

        d2cx2_dx2 = (2*(-2 + 3*x*(2 + (-2 + x)*x))*f0
                     -6*x**3*f1 +
                     2*(g0 + h0 + Y0) +
                     (-1 + x)*x*(6*g0 + 6*h0 + 6*Y0 + (-4 + 8*x - 6*x**2)*df0_dx +
                                 2*(dg0_dx + dh0_dx + dY0_dx) +
                                 x*(6*x*df1_dx - 4*dg0_dx - 4*dh0_dx - 4*dY0_dx +
                                    (-1 + x)*((-2 + 3*x)*d2f0_dx2 - 3*x*d2f1_dx2 +
                                              d2g0_dx2 + d2h0_dx2 + d2Y0_dx2))))/ \
                    (3*(-1 + x)**3*x**3)
        d2cx2_dy2 = ((-2 + 3*x)*d2f0_dy2 - 3*x*d2f1_dy2 + d2g0_dy2 + d2h0_dy2 +
                     d2Y0_dx2)/ \
                    (3*(-1 + x)*x)
        d2cx2_dz2 = ((-2 + 3*x)*d2f0_dz2 - 3*x*d2f1_dz2 + d2g0_dz2 + d2h0_dz2 +
                     d2Y0_dz2)/ \
                    (3*(-1 + x)*x)
        d2cx2_dt2 = ((-2 + 3*x)*d2f0_dt2 - 3*x*d2f1_dt2 + d2g0_dt2 + d2h0_dt2 +
                     d2Y0_dt2)/ \
                    (3*(-1 + x)*x)

        d2cy1_dx2 = -(d2f0_dx2 + (-2 + 3*y**2)*d2g0_dx2 - 3*y**2*d2g1_dx2 +
                      d2h0_dx2 + d2Y0_dx2)/ \
                    (3*(-1 + y)*y)
        d2cy1_dy2 = ((-2 - 6*(-1 + y)*y)*f0 +
                     2*(2 - 3*y*(2 + (-2 + y)*y))*g0 + 6*y**3*g1 -
                     2*(h0 + Y0) - (-1 + y)*y*
                     (6*h0 + 6*Y0 + (2 - 4*y)*df0_dy +
                      (-4 + 8*y - 6*y**2)*dg0_dy + 2*
                      (dh0_dy + dY0_dy) +
                      y*(6*y*dg1_dy - 4*dh0_dy - 4*dY0_dy + (-1 + y)*
                         (d2f0_dy2 + (-2 + 3*y**2)*d2g0_dy2 - 3*y**2*d2g1_dy2 +
                         d2h0_dy2 + d2Y0_dy2))))/ \
                    (3*(-1 + y)**3*y**3)
        d2cy1_dz2 = -(d2f0_dz2 + (-2 + 3*y**2)*d2g0_dz2 - 3*y**2*d2g1_dz2 +
                      d2h0_dz2 + d2Y0_dz2)/ \
                    (3*(-1 + y)*y)
        d2cy1_dt2 = -(d2f0_dt2 + (-2 + 3*y**2)*d2g0_dt2 - 3*y**2*d2g1_dt2 +
                      d2h0_dt2 + d2Y0_dt2)/ \
                    (3*(-1 + y)*y)

        d2cy2_dx2 = (d2f0_dx2 + (-2 + 3*y)*d2g0_dx2 - 3*y*d2g1_dx2 +
                     d2h0_dx2 + d2Y0_dx2)/ \
                    (3*(-1 + y)*y)
        d2cy2_dy2 = ((2 + 6*(-1 + y)*y)*f0 + 2*(-2 + 3*y*(2 + (-2 + y)*y))*g0 -
                     6*y**3*g1 + 2*(h0 + Y0) + (-1 + y)*y*
                     (6*h0 + 6*Y0 + (2 - 4*y)*df0_dy + (-4 + 8*y - 6*y**2)*dg0_dy +
                     2*(dh0_dy + dY0_dy) +
                     y*(6*y*dg1_dy - 4*dh0_dy - 4*dY0_dy +
                       (-1 + y)*(d2f0_dy2 + (-2 + 3*y)*d2g0_dy2 - 3*y*d2g1_dy2 +
                                 d2h0_dy2 + d2Y0_dy2))))/ \
                    (3*(-1 + y)**3*y**3)
        d2cy2_dz2 = (d2f0_dz2 + (-2 + 3*y)*d2g0_dz2 - 3*y*d2g1_dz2 +
                     d2h0_dz2 + d2Y0_dz2)/ \
                    (3*(-1 + y)*y)
        d2cy2_dt2 = (d2f0_dt2 + (-2 + 3*y)*d2g0_dt2 - 3*y*d2g1_dt2 +
                     d2h0_dt2 + d2Y0_dt2)/ \
                    (3*(-1 + y)*y)

        d2cz1_dx2 = -(4*dg0_dx*dh0_dx + d2f0_dx2 + 2*h0*d2g0_dx2 +
                      (3*z**2 + 2*g0)*d2h0_dx2 - 3*z**2*d2h1_dx2 + d2Y0_dx2)/ \
                    (3*(-1 + z)*z)
        d2cz1_dy2 = -(4*dg0_dy*dh0_dy + d2f0_dy2 + 2*h0*d2g0_dy2 +
                      (3*z**2 + 2*g0)*d2h0_dy2 - 3*z**2*d2h1_dy2 + d2Y0_dy2)/ \
                    (3*(-1 + z)*z)
        d2cz1_dz2 = ((-2 - 6*(-1 + z)*z)*f0 + 6*z**3*h1 - 2*Y0 + 2*h0*
                     (-3*z**3 + (-2 - 6*(-1 + z)*z)*g0 + (-1 + z)*z*
                      ((-2 + 4*z)*dg0_dz - (-1 + z)*z*d2g0_dz2)) + (-1 + z)*z*
                      (-6*Y0 + (-2 + 4*z)*df0_dz + (6*z**2 + (-4 + 8*z)*g0 -
                                                    4*(-1 + z)*z*dg0_dz)*dh0_dz -
                       2*dY0_dz + z*
                       (-6*z*dh1_dz + 4*dY0_dz - (-1 + z)*
                        (d2f0_dz2 + (3*z**2 + 2*g0)*d2h0_dz2 - 3*z**2*d2h1_dz2 +
                        d2Y0_dz2))))/ \
                    (3*(-1 + z)**3*z**3)
        d2cz1_dt2 = -(4*dg0_dt*dh0_dy + d2f0_dt2 + 2*h0*d2g0_dt2 +
                      (3*z**2 + 2*g0)*d2h0_dt2 - 3*z**2*d2h1_dt2 + d2Y0_dt2)/ \
                    (3*(-1 + z)*z)

        d2cz2_dx2 = (d2f0_dx2 + d2g0_dx2 + (-2 + 3*z)*d2h0_dx2 -
                     3*z*d2h1_dx2 + d2Y0_dx2)/ \
                    (3*(-1 + z)*z)
        d2cz2_dy2 = (d2f0_dy2 + d2g0_dy2 + (-2 + 3*z)*d2h0_dy2 -
                     3*z*d2h1_dy2 + d2Y0_dy2)/ \
                    (3*(-1 + z)*z)
        d2cz2_dz2 = ((2 + 6*(-1 + z)*z)*f0 + (2 + 6*(-1 + z)*z)*g0 -
                     4*h0 + 2*Y0 + z*
                     (6*(2 + (-2 + z)*z)*h0 - 6*z**2*h1 + (-1 + z)*
                      (6*Y0 + (2 - 4*z)*df0_dz + (2 - 4*z)*dg0_dz -
                      4*dh0_dz + 2*dY0_dz + z*
                      ((8 - 6*z)*dh0_dz + 6*z*dh1_dz - 4*dY0_dz + (-1 + z)*
                       (d2f0_dz2 + d2g0_dz2 + (-2 + 3*z)*d2h0_dz2 -
                        3*z*d2h1_dz2 + d2Y0_dz2)))))/ \
                    (3*(-1 + z)**3*z**3)
        d2cz2_dt2 = (d2f0_dt2 + d2g0_dt2 + (-2 + 3*z)*d2h0_dt2 - 3*z*d2h1_dt2 +
                     d2Y0_dt2)/ \
                    (3*(-1 + z)*z)

        d2ct1_dx2 = (d2f0_dx2 + d2g0_dx2 + d2h0_dx2 - 2*d2Y0_dx2)/(3*t)
        d2ct1_dy2 = (d2f0_dy2 + d2g0_dy2 + d2h0_dy2 - 2*d2Y0_dy2)/(3*t)
        d2ct1_dz2 = (d2f0_dz2 + d2g0_dz2 + d2h0_dz2 - 2*d2Y0_dz2)/(3*t)
        d2ct1_dt2 = (2*f0 + 2*g0 + 2*h0 - 4*Y0 + t*
                     (-2*df0_dt - 2*dg0_dt - 2*dh0_dt + 4*dY0_dt + t*
                      (d2f0_dt2 + d2g0_dt2 + d2h0_dt2 - 2*d2Y0_dt2)))/ \
                    (3*t**3)

        del2c = [
            [d2cx1_dx2, d2cx1_dy2, d2cx1_dz2, d2cx1_dt2],
            [d2cx2_dx2, d2cx2_dy2, d2cx2_dz2, d2cx2_dt2],
            [d2cy1_dx2, d2cy1_dy2, d2cy1_dz2, d2cy1_dt2],
            [d2cy2_dx2, d2cy2_dy2, d2cy2_dz2, d2cy2_dt2],
            [d2cz1_dx2, d2cz1_dy2, d2cz1_dz2, d2cz1_dt2],
            [d2cz2_dx2, d2cz2_dy2, d2cz2_dz2, d2cz2_dt2],
            [d2ct1_dx2, d2ct1_dy2, d2ct1_dz2, d2ct1_dt2]
            ]
        return del2c

    def Pf(self, xyzt):
        """Network coefficient function for 3D diffusion problems"""
        (x, y, z, t) = xyzt
        P = x*(1 - x)*y*(1 - y)*z*(1 - z)*t
        return P

    def delPf(self, xyzt):
        """Network coefficient function gradient"""
        (x, y, z, t) = xyzt
        dP_dx = (1 - 2*x)*y*(1 - y)*z*(1 - z)*t
        dP_dy = x*(1 - x)*(1 - 2*y)*z*(1 - z)*t
        dP_dz = x*(1 - x)*y*(1 - y)*(1 - 2*z)*t
        dP_dt = x*(1 - x)*y*(1 - y)*z*(1 - z)
        delP = [dP_dx, dP_dy, dP_dz, dP_dt]
        return delP

    def del2Pf(self, xyzt):
        """Network coefficient function Laplacian"""
        (x, y, z, t) = xyzt
        d2P_dx2 = -2*y*(1 - y)*z*(1 - z)*t
        d2P_dy2 = -2*x*(1 - x)*z*(1 - z)*t
        d2P_dz2 = -2*x*(1 - x)*y*(1 - y)*t
        d2P_dt2 = 0
        del2P = [d2P_dx2, d2P_dy2, d2P_dz2, d2P_dt2]
        return del2P

#################


# Self-test code

from math import pi, sin, cos

if __name__ == '__main__':

    # Test boundary conditions
    f0f = lambda xyzt: 0
    f1f = lambda xyzt: 0
    g0f = lambda xyzt: 0
    g1f = lambda xyzt: 0
    h0f = lambda xyzt: 0
    h1f = lambda xyzt: 0
    Y0f = lambda xyzt: sin(pi*xyzt[0])*sin(pi*xyzt[1])*sin(pi*xyzt[2])/3
    Y1f = lambda xyzt: None
    bcf = [[f0f, f1f],
           [g0f, g1f],
           [h0f, h1f],
           [Y0f, Y1f]]

    # Test BC gradient
    df0_dxf = lambda xyzt: 0
    df0_dyf = lambda xyzt: 0
    df0_dzf = lambda xyzt: 0
    df0_dtf = lambda xyzt: 0
    df1_dxf = lambda xyzt: 0
    df1_dyf = lambda xyzt: 0
    df1_dzf = lambda xyzt: 0
    df1_dtf = lambda xyzt: 0
    dg0_dxf = lambda xyzt: 0
    dg0_dyf = lambda xyzt: 0
    dg0_dzf = lambda xyzt: 0
    dg0_dtf = lambda xyzt: 0
    dg1_dxf = lambda xyzt: 0
    dg1_dyf = lambda xyzt: 0
    dg1_dzf = lambda xyzt: 0
    dg1_dtf = lambda xyzt: 0
    dh0_dxf = lambda xyzt: 0
    dh0_dyf = lambda xyzt: 0
    dh0_dzf = lambda xyzt: 0
    dh0_dtf = lambda xyzt: 0
    dh1_dxf = lambda xyzt: 0
    dh1_dyf = lambda xyzt: 0
    dh1_dzf = lambda xyzt: 0
    dh1_dtf = lambda xyzt: 0
    dY0_dxf = lambda xyzt: pi*cos(pi*xyzt[0])*sin(pi*xyzt[1])*sin(pi*xyzt[2])/3
    dY0_dyf = lambda xyzt: pi*sin(pi*xyzt[0])*cos(pi*xyzt[1])*sin(pi*xyzt[2])/3
    dY0_dzf = lambda xyzt: pi*sin(pi*xyzt[0])*sin(pi*xyzt[1])*cos(pi*xyzt[2])/3
    dY0_dtf = lambda xyzt: 0
    dY1_dxf = lambda xyzt: None
    dY1_dyf = lambda xyzt: None
    dY1_dzf = lambda xyzt: None
    dY1_dtf = lambda xyzt: None
    delbcf = [[[df0_dxf, df0_dyf, df0_dzf, df0_dtf], [df1_dxf, df1_dyf, df1_dzf, df1_dtf]],
              [[dg0_dxf, dg0_dyf, dg0_dzf, dg0_dtf], [dg1_dxf, dg1_dyf, dg1_dzf, dg1_dtf]],
              [[dh0_dxf, dh0_dyf, dh0_dzf, dh0_dtf], [dh1_dxf, dh1_dyf, dh1_dzf, dh1_dtf]],
              [[dY0_dxf, dY0_dyf, dY0_dzf, dY0_dtf], [dY1_dxf, dY1_dyf, dY1_dzf, dY1_dtf]]]

    # Test BC Laplacian
    d2f0_dx2f = lambda xyzt: 0
    d2f0_dy2f = lambda xyzt: 0
    d2f0_dz2f = lambda xyzt: 0
    d2f0_dt2f = lambda xyzt: 0
    d2f1_dx2f = lambda xyzt: 0
    d2f1_dy2f = lambda xyzt: 0
    d2f1_dz2f = lambda xyzt: 0
    d2f1_dt2f = lambda xyzt: 0
    d2g0_dx2f = lambda xyzt: 0
    d2g0_dy2f = lambda xyzt: 0
    d2g0_dz2f = lambda xyzt: 0
    d2g0_dt2f = lambda xyzt: 0
    d2g1_dx2f = lambda xyzt: 0
    d2g1_dy2f = lambda xyzt: 0
    d2g1_dz2f = lambda xyzt: 0
    d2g1_dt2f = lambda xyzt: 0
    d2h0_dx2f = lambda xyzt: 0
    d2h0_dy2f = lambda xyzt: 0
    d2h0_dz2f = lambda xyzt: 0
    d2h0_dt2f = lambda xyzt: 0
    d2h1_dx2f = lambda xyzt: 0
    d2h1_dy2f = lambda xyzt: 0
    d2h1_dz2f = lambda xyzt: 0
    d2h1_dt2f = lambda xyzt: 0
    d2Y0_dx2f = lambda xyzt: -pi**2*sin(pi*xyzt[0])*sin(pi*xyzt[1])*sin(pi*xyzt[2])/3
    d2Y0_dy2f = lambda xyzt: -pi**2*sin(pi*xyzt[0])*sin(pi*xyzt[1])*sin(pi*xyzt[2])/3
    d2Y0_dz2f = lambda xyzt: -pi**2*sin(pi*xyzt[0])*sin(pi*xyzt[1])*sin(pi*xyzt[2])/3
    d2Y0_dt2f = lambda xyzt: 0
    d2Y1_dx2f = lambda xyzt: None
    d2Y1_dy2f = lambda xyzt: None
    d2Y1_dz2f = lambda xyzt: None
    d2Y1_dt2f = lambda xyzt: None
    del2bcf = [[[d2f0_dx2f, d2f0_dy2f, d2f0_dz2f, d2f0_dt2f], [d2f1_dx2f, d2f1_dy2f, d2f1_dz2f, d2f1_dt2f]],
               [[d2g0_dx2f, d2g0_dy2f, d2g0_dz2f, d2g0_dt2f], [d2g1_dx2f, d2g1_dy2f, d2g1_dz2f, d2g1_dt2f]],
               [[d2h0_dx2f, d2h0_dy2f, d2h0_dz2f, d2h0_dt2f], [d2h1_dx2f, d2h1_dy2f, d2h1_dz2f, d2h1_dt2f]],
               [[d2Y0_dx2f, d2Y0_dy2f, d2Y0_dz2f, d2Y0_dt2f], [d2Y1_dx2f, d2Y1_dy2f, d2Y1_dz2f, d2Y1_dt2f]]]

    # Reference values for tests.
    bc_ref = [[0, 0],
              [0, 0],
              [0, 0],
              [0.301503, None]]
    delbc_ref = [[[0, 0, 0, 0], [0, 0, 0, 0]],
                 [[0, 0, 0, 0], [0, 0, 0, 0]],
                 [[0, 0, 0, 0], [0, 0, 0, 0]],
                 [[0.307764, 0, -0.307764, 0], [None, None, None, None]]]
    del2bc_ref = [[[ 0,        0,        0,       0], [0,    0,    0,    0]],
                  [[ 0,        0,        0,       0], [0,    0,    0,    0]],
                  [[ 0,        0,        0,       0], [0,    0,    0,    0]],
                  [[-2.97571, -2.97571, -2.97571, 0], [None, None, None, None]]]
    c_ref = [0.418754, -0.418754, 0.402004, -0.402004, 0.418754,
             -0.418754, -0.287146]
    delc_ref = [[ 0.0784879, 0, -0.427449,  0],
                [-0.0784879, 0,  0.427449,  0],
                [ 0.410352,  0, -0.410352,  0],
                [-0.410352,  0,  0.410352,  0],
                [ 0.427449,  0, -0.0784879, 0],
                [-0.427449,  0,  0.0784879, 0],
                [-0.293108,  0,  0.293108, 0.410208]]
    del2c_ref = [[-0.774133, -4.13294,  -4.13294,  0],
                 [ 0.774133,  4.13294,   4.13294,  0],
                 [-3.96762,  -0.751588, -3.96762,  0],
                 [ 3.96762,   0.751588,  3.96762,  0],
                 [-4.13294,  -4.13294,  -0.774133, 0],
                 [ 4.13294,   4.13294,   0.774133, 0],
                 [ 2.83401,   2.83401,   2.83401, -1.17202]]
    A_ref = 0.100501
    delA_ref = [0.102588, 0, -0.102588, 0]
    del2A_ref = [-0.991905, -0.991905, -0.991905, 0]
    P_ref = 0.01008
    delP_ref = [0.0084, 0.0, -0.0084, 0.0144]
    del2P_ref = [-0.084, -0.08064, -0.084, 0]
    Yt_ref = 0.101741
    delYt_ref = [0.104629, 0.002016, -0.100597, 0.0058032]
    del2Yt_ref = [-0.999448, -0.999606, -1.00395, 0.0159552]

    # # Additional test variables.
    N_ref = 0.123
    delN_ref = [0.1, 0.2, 0.3, 0.4]
    del2N_ref = [0.11, 0.22, 0.33, 0.44]

    # Test all functions near the center of the domain.
    xyzt = [0.4, 0.5, 0.6, 0.7]

    # Create a new trial function object.
    tf = Diff3DTrialFunction(bcf, delbcf, del2bcf)

    print("Testing boundary conditions.")
    for i in range(len(tf.bcf)):
        for (j, f) in enumerate(tf.bcf[i]):
            bc = f(xyzt)
            if ((bc_ref[i][j] is not None and not np.isclose(bc, bc_ref[i][j]))
                or (bc_ref[i][j] is None and bc is not None)):
                print("ERROR: bc[%d][%d] = %s, vs ref %s" % (i, j, bc, bc_ref[i][j]))

    print("Testing boundary condition gradients.")
    for i in range(len(tf.delbcf)):
        for j in range(len(tf.delbcf[i])):
            for (k, f) in enumerate(tf.delbcf[i][j]):
                delbc = f(xyzt)
                if ((delbc_ref[i][j][k] is not None and not np.isclose(delbc, delbc_ref[i][j][k]))
                    or (delbc_ref[i][j][k] is None and delbc is not None)):
                    print("ERROR: delbc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, delbc, delbc_ref[i][j][k]))

    print("Testing boundary condition Laplacians.")
    for i in range(len(tf.del2bcf)):
        for j in range(len(tf.del2bcf[i])):
            for (k, f) in enumerate(tf.del2bcf[i][j]):
                del2bc = f(xyzt)
                if ((del2bc_ref[i][j][k] is not None and not np.isclose(del2bc, del2bc_ref[i][j][k]))
                    or (del2bc_ref[i][j][k] is None and del2bc is not None)):
                    print("ERROR: del2bc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, del2bc, del2bc_ref[i][j][k]))

    print("Testing coefficients.")
    c = tf.cf(xyzt)
    for (i, cc) in enumerate(c):
        if not np.isclose(cc, c_ref[i]):
            print("ERROR: c[%d] = %s, vs ref %s" % (i, cc, c_ref[i]))

    print("Testing coefficient gradients.")
    delc = tf.delcf(xyzt)
    for i in range(len(c)):
        for (j, delci) in enumerate(delc[i]):
            if not np.isclose(delci, delc_ref[i][j]):
                print("ERROR: delc[%d][%d] = %s, vs ref %s" % (i, j, delci, delc_ref[i][j]))

    print("Testing coefficient Laplacians.")
    del2c = tf.del2cf(xyzt)
    for i in range(len(del2c)):
        for (j, del2ci) in enumerate(del2c[i]):
            if not np.isclose(del2ci, del2c_ref[i][j]):
                print("ERROR: del2c[%d][%d] = %s, vs ref %s" % (i, j, del2ci, del2c_ref[i][j]))

    print("Testing boundary condition function.")
    A = tf.Af(xyzt)
    if not np.isclose(A, A_ref):
        print("ERROR: A = %s, vs ref %s" % (A, A_ref))

    print("Testing boundary condition function gradient.")
    delA = tf.delAf(xyzt)
    for (i, delAi) in enumerate(delA):
        if not np.isclose(delAi, delA_ref[i]):
            print("ERROR: delA[%d] = %s, vs ref %s" % (i, delAi, delA_ref[i]))

    print("Testing boundary condition function Laplacian.")
    del2A = tf.del2Af(xyzt)
    for (i, del2Ai) in enumerate(del2A):
        if not np.isclose(del2Ai, del2A_ref[i]):
            print("ERROR: del2A[%d] = %s, vs ref %s" % (i, del2Ai, del2A_ref[i]))

    print("Testing network coefficient function.")
    P = tf.Pf(xyzt)
    if not np.isclose(P, P_ref):
        print("ERROR: P = %s, vs ref %s" % (P, P_ref))

    print("Testing network coefficient function gradient.")
    delP = tf.delPf(xyzt)
    for (i, delPi) in enumerate(delP):
        if not np.isclose(delPi, delP_ref[i]):
            print("ERROR: delP[%d] = %s, vs ref %s" % (i, delPi, delP_ref[i]))

    print("Testing network coefficient function Laplacian.")
    del2P = tf.del2Pf(xyzt)
    for (i, del2Pi) in enumerate(del2P):
        if not np.isclose(del2Pi, del2P_ref[i]):
            print("ERROR: del2P[%d] = %s, vs ref %s" % (i, del2Pi, del2P_ref[i]))

    print("Testing trial function.")
    Yt = tf.Ytf(xyzt, N_ref)
    if not np.isclose(Yt, Yt_ref):
        print("ERROR: Yt = %s, vs ref %s" % (Yt, Yt_ref))

    print("Testing trial function gradient.")
    delYt = tf.delYtf(xyzt, N_ref, delN_ref)
    for (i, delYti) in enumerate(delYt):
        if not np.isclose(delYti, delYt_ref[i]):
            print("ERROR: delYt[%d] = %s, vs ref %s" % (i, delYti, delYt_ref[i]))

    print("Testing trial function Laplacian.")
    del2Yt = tf.del2Ytf(xyzt, N_ref, delN_ref, del2N_ref)
    for (i, del2Yti) in enumerate(del2Yt):
        if not np.isclose(del2Yti, del2Yt_ref[i]):
            print("ERROR: del2Yt[%d] = %s, vs ref %s" % (i, del2Yti, del2Yt_ref[i]))
