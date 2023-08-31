import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
import random
import copy
from astropy.convolution import convolve, Gaussian1DKernel
def Moke_function(x_array, amp1, cen1, sigma1, amp2, cen2, sigma2):

    return 1*(np.exp(-amp1*np.exp(-(x_array-cen1)**2*4*np.log(2)/(sigma1**2)))) + \
        1 * (np.exp(-amp2 * np.exp(-(x_array - cen2) ** 2 * 4 * np.log(2) / (sigma2 ** 2)))) - 1

cen1 = 2796
cen2 = 2803
amp1 = 6
amp2 = amp1/2
sigma1 = sigma2 = 2
x_array = np.arange(cen1-7, cen2+7, 0.01)
parameterlist = [amp1, cen1, sigma1, amp2, cen2, sigma2]
lati = []
noise = np.linspace(0.2, 0.5, len(x_array))

for i in range(len(x_array)):
    lati.append(np.random.uniform(-1, 1))

def integration(x_array, parameterlist):
    """
    use legendre methods to calculate the integration of
    double gaussian function

    """
    amp1, cen1, sigma1, amp2, cen2, sigma2 = parameterlist
    N = 100  # select 100 sample points
    down, up = min(x_array), max(x_array)  # the range of integration
    x, wi = roots_legendre(N)
    xi = x * (up - down) / 2 + (up + down) / 2
    sum = (up - down) / 2 * np.dot(wi, -(Moke_function(xi, amp1, cen1, sigma1, amp2, cen2, sigma2)-1))
    return sum
def calcuequivalent_width(x_array, parameterlist):
    """
    use integration of estimated double gaussian function to
    calculate rest equivalent width of MGII system

    """
    sum = integration(x_array, parameterlist)
    equw = sum/1
    # continuum are already normalized to 1
    return equw
equw = calcuequivalent_width(x_array, parameterlist)


Moke = Moke_function(x_array, amp1, cen1, sigma1, amp2, cen2, sigma2)
origin = copy.copy(Moke)
for j in range(int(len(Moke))):
    Moke[j] = Moke[j]+(lati[j]*noise[j])
Moke1 = convolve(Moke, Gaussian1DKernel(2))
plt.figure(figsize=(12, 6))
plt.plot(x_array, Moke1, label="Moke")
plt.plot(x_array, noise, label="Noise")
plt.legend()
plt.show()
plt.figure(figsize=(12, 6))
plt.plot(x_array, Moke, label="Moke")
plt.plot(x_array, noise, label="Noise")
plt.legend()
plt.show()
plt.figure(figsize=(12, 6))
plt.plot(x_array, origin)
plt.ylim([0, 1.3])

plt.show()