import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
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

    #lam = lam / (z + 1.)
    x = (lam - lam0) / dl_D + 0.0001

    tau = np.float64(C_a) * N * H(a, x)

    return tau
def Voigt_I(lam, lam0, f, N, b, gam):
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

    x = (lam - lam0) / dl_D + 0.0001

    tau = np.float64(C_a) * N * H(a, x)
    Voigt_I = np.exp(-tau)
    return Voigt_I

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
    lam0 = 1215.67
    f = 0.4164
    gam = 6.265e8

    dlam = 0.01
    lam = np.arange(lam0-7, lam0+7, dlam)
    Imatrix = np.zeros((len(N), len(lam)))
    for i, N_i in enumerate(N):
        profile = Voigt(lam, lam0, f, N_i, b, gam)
        I = np.exp(-profile)
        W[i] = np.sum(1.-I)*dlam
        Imatrix[i] = I
    tau_0 = N*f*lam0
    return (tau_0, 10e3*W/lam0, N, Imatrix, lam)
def deltalam(Resolution, lam0):
    deltalam = lam0/Resolution
    return deltalam


def ISF(Resolution):
    lam0 = 1215.67
    dlam = 0.01
    lamlam = np.arange(lam0-7, lam0+7, dlam)
    deltalam1 = deltalam(Resolution, lam0)
    delsigma = deltalam1/2.35
    param = 1/(np.sqrt(2*np.pi)*delsigma)
    ISF = param * np.exp(-((lamlam - lam0) ** 2) / (2 * delsigma ** 2))
    cal = int(len(lamlam)/20)
    ISF_matrix = np.zeros((cal, len(lamlam)))
    lambda_value = []
    for n in range(cal):
        # every 0.2 A one pixel
        ISF_matrix[n, :] = param * np.exp(-((lamlam - lamlam[n*20]) ** 2) / (2 * delsigma ** 2))
        lambda_value.append(lamlam[n*20])
    return ISF, lamlam, ISF_matrix, lambda_value, cal

def convolve(b, ISF_matrix, cal,  Nmin=12., Nmax=22, num=100):
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
    lam0 = 1215.67
    f = 0.4164
    gam = 6.265e8

    dlam = 0.01
    lam = np.arange(lam0-7, lam0+7, dlam)
    convolvematrix = np.zeros((len(N), cal))
    for i, N_i in enumerate(N):
        profile = Voigt(lam, lam0, f, N_i, b, gam)
        I = np.exp(-profile)
        for j in range(cal):
            convolve = ISF_matrix[j]*(-(I-1))
            value = integrate.trapz(convolve, lam)
            convolvematrix[i, j] = (-value)+1

    return convolvematrix

def curve_fitting(convolvematrix, lambda_value, b, cal):
    N = np.logspace(12, 21, 100)
    W = np.zeros_like(N)
    b *= 1.e5
    # velocity already suit to cm
    # Define line parameters, they are not important.
    lam0 = 1215.67
    f = 0.4164
    gam = 6.265e8

    lam = lambda_value
    I_fittingmatrix = np.zeros((len(N), cal))
    param_list = np.zeros((len(N), 5))
    for i, N_i in enumerate(N):
        popt_voigt, pcov_voigt = curve_fit(Voigt_I, lambda_value, convolvematrix[i], maxfev=1000000000, p0=[lam0, f, N_i, b, gam])
        Voigt_fitting = Voigt(lambda_value, *popt_voigt)
        I_fitting = np.exp(-Voigt_fitting)
        I_fittingmatrix[i] = I_fitting
        param_list[i] = popt_voigt
    return I_fittingmatrix, param_list

Resolution1 = 450
Resolution2 = 2250
Resolution3 = 4500
Resolution4 = 9000
Resolution5 = 20000
ISF1, lamlam1, ISF_matrix1, lambda_value1, cal1 = ISF(Resolution1)
ISF2, lamlam2, ISF_matrix2, lambda_value2, cal2 = ISF(Resolution2)
ISF3, lamlam3, ISF_matrix3, lambda_value3, cal3 = ISF(Resolution3)
ISF4, lamlam4, ISF_matrix4, lambda_value4, cal4 = ISF(Resolution4)
ISF5, lamlam5, ISF_matrix5, lambda_value5, cal5 = ISF(Resolution5)

b = 50
convolvematrix = convolve(b, ISF_matrix5, cal5, Nmin=12., Nmax=21, num=100)
I_fittingmatrix, param_list = curve_fitting(convolvematrix, lambda_value1, b, cal1)
x, y, N, I, lam = curve_of_growth(b, Nmin=12., Nmax=21, num=100)
b2 = 30
x1, y1, N1, I1, lam1 = curve_of_growth(b2, Nmin=12., Nmax=21, num=100)
b3 = 100
x2, y2, N2, I2, lam2 = curve_of_growth(b3, Nmin=12., Nmax=21, num=100)
print("b1=",param_list[20, 3]/10e4, "b2=",param_list[70, 3]/10e4)

#fig = plt.figure(figsize=(6, 5))
#ax1 = fig.add_subplot(1, 1, 1)
#ax1.plot(lamlam1, ISF1, 'g', label = 'R=450')
#ax1.plot(lamlam2, ISF2, 'g', label = 'R=2250')
#ax1.plot(lamlam3, ISF3, 'g', label = 'R=4500')
#ax1.plot(lamlam4, ISF4, 'g', label = 'R=9000')
#ax1.plot(lamlam5, ISF5, 'g', label = 'R=20000')
#ax1.set_xlabel("$\Delta$$\lambda$")
#ax1.set_ylabel("Rest EW(Ang)")
#plt.legend()
#plt.show()
#ax1.legend()
#for i in range(10):
    #print(np.log10(N[10*i]))
#plt.show()



fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(1, 1, 1)
#ax1.plot(lambda_value1, convolvematrix[0])
#ax1.plot(lambda_value1, convolvematrix[10])
ax1.step(lambda_value1, convolvematrix[20], label ='raw1\nlogN = 13.81cm$^{-2}$')
ax1.plot(lambda_value1, I_fittingmatrix[20], label ='Voigt fitting1\nlogN = 13.81cm$^{-2}$')
#ax1.plot(lambda_value1, convolvematrix[30])
#ax1.plot(lambda_value1, convolvematrix[40])
#ax1.plot(lambda_value1, convolvematrix[50])
#ax1.plot(lambda_value1, convolvematrix[60])
ax1.step(lambda_value1, convolvematrix[70], label ='raw2\nlogN = 18.36cm$^{-2}$')
ax1.plot(lambda_value1, I_fittingmatrix[70], label ='Voigt fitting2\nlogN = 18.36cm$^{-2}$')
#ax1.plot(lambda_value1, convolvematrix[80], label = 'logN = 19.27cm$^{-2}$')
#ax1.plot(lambda_value1, convolvematrix[90])
ax1.set_ylabel("Residual flux")
ax1.set_xlabel("Wavelength")
#plt.xlim([1213.67, 1217.67])
plt.ylim([0, 1])
plt.legend()
plt.show()

fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(1, 1, 1)
#ax1.plot(lam, I[0])
#ax1.plot(lam, I[10])
ax1.plot(lam, I[20], label = 'logN = 13.81cm$^{-2}$')
#ax1.plot(lam, I[30])
#ax1.plot(lam, I[40])
#ax1.plot(lam, I[50])
#ax1.plot(lam, I[60])
ax1.plot(lam, I[70], label = 'logN = 18.36cm$^{-2}$')
#ax1.plot(lam, I[80], label = 'logN = 19.27cm$^{-2}$')
#ax1.plot(lam, I[90])
ax1.set_ylabel("Residual flux")
ax1.set_xlabel("Wavelength")
#plt.xlim([1213.67, 1217.67])
plt.ylim([0, 1])
plt.legend()
plt.show()