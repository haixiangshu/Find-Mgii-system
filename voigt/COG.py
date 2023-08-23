import numpy as np
import matplotlib.pyplot as plt

# ==== PARAMETERS ==================

c = 2.99792e10		# cm/s
m_e = 9.1095e-28		# g
e = 4.8032e-10		# cgs units


# ==== VOIGT PROFILE ===============
def H(a, x):
    P = x**2
    H0 = np.exp(-x**2)
    Q = 1.5/x**2
    return H0 - a/np.sqrt(np.pi)/P * (H0*H0*(4.*P*P + 7.*P + 4. + Q) - Q - 1)


def Voigt(lam, lam0, f, N, b, gam, z=0):
    """Calculate the Voigt profile of transition with
    rest frame transition wavelength: 'l0'
    oscillator strength: 'f'
    column density: N  cm^-2
    velocity width: b  cm/s
    """

    # ==================================
    # Calculate Profile
    dl_D = b / c * lam0

    C_a = np.sqrt(np.pi) * e**2 * f * (lam0 * 1.e-8) / m_e / c / b
    a = (lam0*1.e-8) * gam / (4.*np.pi * b)

    lam = lam / (z + 1.)
    x = (lam - lam0) / dl_D + 0.0001

    tau = np.float64(C_a) * N * H(a, x)

    return tau


def curve_of_growth(b, Nmin=12., Nmax=22, num=100):
    """

    Return the Curve of Growth for a Voigt line profile.
    b is given in km/s.

    Returns:

    tau_0 : N*f*l0, proportional to optical depth at line core
    W : rest equivalent width divided by transition wavelength, l0

    """
    N = np.logspace(Nmin, Nmax, num)
    W = np.zeros_like(N)
    b *= 1.e5
    # velocity already suit to cm
    # Define line parameters, they are not important.
    lam0 = 2803.531
    f = 0.303
    gam = 2.57e8

    dlam = 0.01
    lam = np.arange(lam0-10, lam0+10, dlam)
    Imatrix = np.zeros((len(N), len(lam)))
    for i, N_i in enumerate(N):
        profile = Voigt(lam, lam0, f, N_i, b, gam)
        I = np.exp(-profile)
        W[i] = np.sum(1.-I)*dlam
        Imatrix[i] = I
    tau_0 = N*f*lam0
    return (tau_0, 10e3*W/lam0, N, Imatrix, lam)
b = 10
x, y, N, I, lam = curve_of_growth(b, Nmin=12., Nmax=21, num=100)
b2 = 30
x1, y1, N1, I1, lam1 = curve_of_growth(b2, Nmin=12., Nmax=21, num=100)
b3 = 100
x2, y2, N2, I2, lam2 = curve_of_growth(b3, Nmin=12., Nmax=21, num=100)
fig = plt.figure(figsize=(6, 5))
ax1 = fig.add_subplot(1, 1, 1)
ax1.loglog(N, y, label = 'b=10km/s')
ax1.loglog(N1, y1, label = 'b=30km/s')
ax1.loglog(N2, y2, label = 'b=100km/s')
ax1.set_xlabel("N$_{MgII}$")
ax1.set_ylabel("Rest EW(Ang)")
plt.legend()
plt.show()
#ax1.legend()
#for i in range(10):
    #print(np.log10(N[10*i]))
plt.show()



fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(lam, I[0], label = 'logN = 12.00cm$^{-2}$')
ax1.plot(lam, I[10], label = 'logN = 12.91cm$^{-2}$')
ax1.plot(lam, I[20], label = 'logN = 13.81cm$^{-2}$')
ax1.plot(lam, I[30], label = 'logN = 14.72cm$^{-2}$')
ax1.plot(lam, I[40], label = 'logN = 15.63cm$^{-2}$')
ax1.plot(lam, I[50], label = 'logN = 16.54cm$^{-2}$')
ax1.plot(lam, I[60], label = 'logN = 17.45cm$^{-2}$')
ax1.plot(lam, I[70], label = 'logN = 18.36cm$^{-2}$')
ax1.plot(lam, I[80], label = 'logN = 19.27cm$^{-2}$')
ax1.plot(lam, I[90], label = 'logN = 20.18cm$^{-2}$')
ax1.set_ylabel("Residual flux")
ax1.set_xlabel("Wavelength")
#plt.xlim([1213.67, 1217.67])
plt.ylim([0, 1])
plt.legend()
plt.show()