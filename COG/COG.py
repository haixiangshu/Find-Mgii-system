import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
# ==== PARAMETERS ==================

c = 2.99792e8		# m/s
m_e = 9.1095e-28		# g
e = 4.8032e-10		# cgs units
lambda_0 = 1215.67e-10  # m/s

# ==== GAUSSIAN PROFILE ===============

b = 10e3
b1 = 20e3
b2 = 30e3
b3 = 30e4
def HWMH(b):
    """
        HWMH of gaussian
    """
    g = np.sqrt(np.log(2))*b*lambda_0/c
    return g

def tau_gaussian(x, tau_0, g):
    """
        single gaussian function
    """
    tau_gaussian = []
    for i in range(len(tau_0)):
        tau_gaussian.append(np.exp(-(tau_0[i]*np.exp(-(x**2)*np.log(2)/(g**2)))))
    return tau_gaussian
def tau_gaussian1dimention(x, tau_0, g):
    """
        single gaussian function
    """
    tau_gaussian = np.exp(-(tau_0*np.exp(-(x**2)*np.log(2)/(g**2))))
    return tau_gaussian
def x_array(wavelength, lambda_0):
    x = wavelength-lambda_0
    return x
wavelength = np.linspace(1214e-10, 1216e-10, 1000)

x = x_array(wavelength, lambda_0)

tau_0 = [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8]
g = HWMH(b)
g1 = HWMH(b1)
g2 = HWMH(b2)
g3 = HWMH(b3)
tau_gaussian = tau_gaussian(x, tau_0, g)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x*np.sqrt(np.log(2))/g, tau_gaussian[0], label=r'$\tau$(0)=1/16')
ax1.plot(x*np.sqrt(np.log(2))/g, tau_gaussian[1], label=r'$\tau$(0)=1/8')
ax1.plot(x*np.sqrt(np.log(2))/g, tau_gaussian[2], label=r'$\tau$(0)=1/4')
ax1.plot(x*np.sqrt(np.log(2))/g, tau_gaussian[3], label=r'$\tau$(0)=1/2')
ax1.plot(x*np.sqrt(np.log(2))/g, tau_gaussian[4], label=r'$\tau$(0)=1')
ax1.plot(x*np.sqrt(np.log(2))/g, tau_gaussian[5], label=r'$\tau$(0)=2')
ax1.plot(x*np.sqrt(np.log(2))/g, tau_gaussian[6], label=r'$\tau$(0)=4')
ax1.plot(x*np.sqrt(np.log(2))/g, tau_gaussian[7], label=r'$\tau$(0)=8')
plt.xlabel("x$\sqrt{ln2}$/g")
plt.ylabel("Intensity")
plt.ylim([0, 1])
plt.xlim([-4, 4])
plt.legend()
plt.show()

def integration(x, tau_0, g):
    """
    use legendre methods to calculate the integration of
    double gaussian function

    """
    sum = []
    N = 100  # select 100 sample points
    down, up = min(x), max(x)  # the range of integration
    x, wi = roots_legendre(N)
    xi = x * (up - down) / 2 + (up + down) / 2
    for i in range(len(tau_0)):
        sum.append((up - down) / 2 * np.dot(wi, -(tau_gaussian1dimention(xi, tau_0[i], g)-1)))
    return sum
EQUV = integration(x, tau_0, g)
EQUV1 = integration(x, tau_0, g1)
EQUV2 = integration(x, tau_0, g2)
EQUV3 = integration(x, tau_0, g3)
def tau_lambda0(tau_gaussian):
    tau_lambda0 = []
    for i in range(len(tau_gaussian)):
        tau_lambda0.append(-np.log(min(tau_gaussian[i])))
    return tau_lambda0
tau_lambda0 = tau_lambda0(tau_gaussian)


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(np.log10(tau_lambda0), np.log10(EQUV), label='b = 10 km/s')
ax1.plot(np.log10(tau_lambda0), np.log10(EQUV1), label='b = 20 km/s')
ax1.plot(np.log10(tau_lambda0), np.log10(EQUV2), label='b = 30 km/s')
ax1.plot(np.log10(tau_lambda0), np.log10(EQUV3), label='b = 40 km/s')
plt.ylabel("log$_{10}$[W/nm]")
plt.xlabel("log$_{10}$$\\tau$($\lambda$$_{0}$)")
#plt.ylim([-10, 10])
#plt.xlim([-10, 10])
plt.legend()
plt.show()