import sys
from astropy.io import fits as pyfits
import numpy as np
from pathlib import Path
import functools
from astropy.convolution import convolve
from sklearn import mixture
from astropy.io import fits
from astropy.table import Table, vstack, hstack, unique, join
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
import os
# sys.path.append("/global/u2/b/bid13/VI/prospect/py")
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.optimize import curve_fit
from scipy.special import roots_legendre
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
cat_path = '/home/haixiangshu/Documents/UVES'
cat_name = 'ADP.2019-03-12T05_26_45.026.fits'
cat_path = os.path.join(cat_path, cat_name)
hdulist = pyfits.open(cat_path)

# print column information
#print(hdulist[1].columns)

# get to the data part (in extension 1)
scidata = hdulist[1].data

wave = scidata[0][0]
normalized_flux = scidata[0][1]
error = scidata[0][2]
continuum = scidata[0][3]
flux = normalized_flux*continuum
error = 1/error**0.5
#suit the frame which error = 1/sigma**2
normalized_error = error/continuum
z = 2.406
def search_window(normalized_flux, normalized_error, wavelength, error, continuum, flux, z):
    """
                param flux;noise;continuum_flux;wavelength,z,normalized_flux, normalized_inv
                define search window
                avoids misidentification due to the cases where the continuum
                is not well-fitted to the quasar’s intrinsic C iv or Mg ii emission lines
    """
    nummin = 0
    nummax = 0
    for i in range(len(flux)):
        #if wavelength[i] > (1+z)*1550:
        if wavelength[i] > 4200:
            nummin = i
            break
    for i in range(len(flux)):
        #if wavelength[i] > (1+z)*2800:
        if wavelength[i] > 8500:
            nummax = i
            break
    normalized_fluxnew = normalized_flux[nummin:nummax]
    normalized_errornew = normalized_error[nummin:nummax]
    wavenew = wavelength[nummin:nummax]
    errornew = error[nummin:nummax]
    continuumnew = continuum[nummin:nummax]
    fluxnew = flux[nummin:nummax]
    return normalized_fluxnew, normalized_errornew, wavenew, errornew, continuumnew, fluxnew
normalized_fluxnew, normalized_errornew, wavenew, errornew, continuumnew, fluxnew = search_window(normalized_flux,normalized_error, wave, error, continuum, flux, z)





def gaussian_filter(normalized_flux, normalized_error):
    """
                    param normalized_flux, normalized_inv
                    use gaussian filter to  convolve the residual and the noise estimates
                    with a width of 3-8 pixels each
    """
    data1 = normalized_flux
    data2 = normalized_error
    normalized_flux = gaussian_filter1d(data1, sigma=1)
    normalized_error = gaussian_filter1d(data2, sigma=1)
    return normalized_flux, normalized_error
normalized_fluxnew, normalized_errornew = gaussian_filter(normalized_fluxnew, normalized_errornew)
normalized_fluxnew = (-(normalized_fluxnew - 1)) + 1
#reverse the flux to suit the "Find peaks"

#plt.plot(wavenew, normalized_fluxnew, 'k', label='normalized_flux')
#plt.plot(wavenew, normalized_errornew, 'r', label='Error')
#plt.ylim([0, 2])
#plt.xlabel('$\lambda$ [$\AA$]')
#plt.ylabel('Residual')
#plt.legend()
#plt.show()

def Criterion_MGII(normalized_flux, normalized_error, wavelength, z):
    """
            param normalizedflux;normalizedinv;;wavelength,z:
            use method:
            SNR(MGII lamda2796)>4 && SNR(MGII lamda2803)>2
            return:location in shift(0~3)
    """
    location1 = np.zeros((len(normalized_flux)))
    location2 = np.zeros((len(normalized_flux)))
    wavemax = 2803 * (1+z)
    observe_wave = wavelength
    for ii in range(len(normalized_flux)):
        if (ii + 3) <= (len(normalized_flux)-1):
            signal = np.absolute(normalized_flux[ii]-1) + np.absolute(normalized_flux[ii+1]-1) + np.absolute(normalized_flux[ii+2]-1) + np.absolute(normalized_flux[ii+3]-1)
            SNR = signal/(normalized_error[ii] + normalized_error[ii+1] + normalized_error[ii+2] + normalized_error[ii+3])
            if SNR > 1 and wavemax > observe_wave[ii]:
                location1[ii:ii+4] = observe_wave[ii:ii+4]
            if SNR > 2 and wavemax > observe_wave[ii]:
                location2[ii:ii+4] = observe_wave[ii:ii+4]
        # SNR number is replaceable, to search for best fitting you need to select by your self
        # the mgii system are observed in front of qso
    return location1, location2

location1, location2 = Criterion_MGII(normalized_fluxnew, normalized_errornew, wavenew, z)


def Find_peaks(normalized_fluxnew, wavelength, final_loc, final_locrow):
    """
    methods to find peaks in the normalized flux
    there are 72 pixels for each wavelength so we choose 576 pixels sumadd
    """

    peakcandi = np.zeros((len(normalized_fluxnew)))
    for i in range(len(final_loc)):
        num = final_loc[i]
        peakcandi[num] = normalized_fluxnew[num]
    peaks, _ = find_peaks(peakcandi, height=1.2, threshold=None, distance=(2803-2796)*72, prominence=None, width=None, wlen=None, rel_height=None, plateau_size=None)

    return peaks
def split_candidates(final_loc, final_locrow, final_loc1, final_locrow1, z, system):
    """
    the specific length of each candidate to split array into specific length subarray
    """

    waverangemax = (2803 - 2796) * (1 + z)
    # we decide range od each mgii peak is less than 5
    for k in range(len(final_loc)):
        if k > 0:
            if np.max(final_locrow) - (np.min(final_locrow) + 10) <= waverangemax:
                # 10 is related to line separation of the absorbers
                final_loc1.append(final_loc[0:len(final_loc)-1])
                final_locrow1.append(final_locrow[0:len(final_loc)-1])
                system.append(len(final_loc)-1)
                return system
            if final_locrow[k] - (np.min(final_locrow) + 10) > waverangemax:
                if 6 > k:
                    for c in range(0, k):
                        final_loc = np.delete(final_loc, 0, None)
                        final_locrow = np.delete(final_locrow, 0, None)
                    break
                if k > 6:
                    system.append(k-1)
                if len(final_loc) > 0:
                    for d in range(0, k):
                        final_loc1.append(final_loc[0])
                        final_locrow1.append(final_locrow[0])
                        final_loc = np.delete(final_loc, 0, None)
                        final_locrow = np.delete(final_locrow, 0, None)
                    break
    return system, final_loc1, final_locrow1, split_candidates(final_loc, final_locrow, final_loc1, final_locrow1, z, system)



# Define the Gaussian function

def _1gaussian(x_array, amp1, cen1, sigma1):
    """
        single gaussian function
    """
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2)))
def _2gaussian(x_array, amp1, cen1, sigma1, amp2, cen2, sigma2):
    """
        double gaussian function
    """
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2))) + \
            amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen2)/sigma2)**2))) + 1
def suit_system(location1, location2, wavelength, z):
    """
                param flux;location1;location2;wavelength,z:
                use method:
                double Gaussian profile to fit doublet candidates
                return:location of final MGII , shift of each system
    """
    loc = []
    locrow = []
    for i in range(len(location1)):
        num1 = i
        if location1[i] > 0:
            loc.append(num1)
            locrow.append(wavelength[num1])
    for j in range(len(location2)):
        num2 = j
        if location2[j] > 0:
            loc.append(num2)
            locrow.append(wavelength[num2])
    loc.sort()
    locrow.sort()
    final_loc = np.unique(loc)
    final_locrow = np.unique(locrow)
    system = []
    final_loc1 = []
    final_locrow1 = []
    peaks = Find_peaks(normalized_fluxnew, wavenew, final_loc, final_locrow)
    system, final_loc1, final_locrow1, d = split_candidates(final_loc, final_locrow, final_loc1, final_locrow1, z, system)
    system = np.array(system)
    return system, final_loc1, final_locrow1, final_loc, final_locrow

def double_Gaussian_profile(location1, location2, flux, wavelength, z):
    #each n repesent one possible mgii system
    system, final_loc1, final_locrow1, final_loc, final_locrow= suit_system(location1, location2, wavelength, z)
    peaks = Find_peaks(normalized_fluxnew, wavenew, final_loc, final_locrow)

    xfinaldata = np.empty(0)
    yfinaldata = np.empty(0)
    parameterlist = []
    coveriancelist = []
    gauss_peak_list = []
    gaps = []

    for k in range(len(peaks)-1):
        #do double gaussian fix
        if k <= len(peaks)-2:
            num1 = peaks[k]
            num2 = peaks[k+1]
            ydata = []
            xdata = np.empty(0)
            location = np.empty(0)
            location = np.append(location, np.arange(num1 - (4 * 72), num1 + (4* 72)))
            location = np.append(location, arange(num2 - (4 * 72), num2 + (4 * 72)))
            xdata = np.append(xdata, wavelength[num1 - (4 * 72): num1 + (4 * 72)])
            xdata = np.append(xdata, wavelength[num2 - (4 * 72): num2 + (4 * 72)])
            location = np.unique(location)
            xdata = np.unique(xdata)
            gaps.append(len(xdata))
            for i in range(len(location)):
                locval = int(location[i])
                ydata.append(flux[locval])
            x_array = np.asarray(xdata)
            y_array_2gauss = np.asarray(ydata)
            center = x_array[72*4]
            center2 = x_array[-72*4]
            sigma = 1
            # Define the Gaussian function
            # we estimate parameter below for reducing calculate
            amp1 = 100
            cen1 = center
            sigma1 = sigma
            amp2 = 80
            cen2 = center2
            sigma2 = sigma
            _2gauss = _2gaussian(x_array, amp1, cen1, sigma1, amp2, cen2, sigma2)
            popt_2gauss, pcov_2gauss = curve_fit(_2gaussian, x_array, y_array_2gauss, maxfev = 1000000000, p0=[amp1, cen1, sigma1, amp2, cen2, sigma2])
            pcov_2gauss = np.sqrt(np.diag(pcov_2gauss))
            gauss_peak_main = _2gaussian(x_array, *popt_2gauss)
            gauss_peak_main = list(gauss_peak_main)
            popt_2gauss = list(popt_2gauss)
            pcov_2gauss = list(pcov_2gauss)
            gauss_peak_list = gauss_peak_list + gauss_peak_main
            parameterlist = parameterlist +popt_2gauss
            coveriancelist = coveriancelist + pcov_2gauss
            xfinaldata = np.append(xfinaldata, x_array)
            yfinaldata = np.append(yfinaldata, y_array_2gauss)

    return xfinaldata, yfinaldata, parameterlist, coveriancelist, gauss_peak_list, gaps


xfinaldata, yfinaldata, parameterlist, coveriancelist, gauss_peak_list, gaps= double_Gaussian_profile(location1, location2, normalized_fluxnew, wavenew, z)
region = []
num1 = 0
cal = 0
for o in range(len(gaps)-1):
    cal = num1
    num1 += gaps[o]
    region.append([cal, num1])
print(region)
#for every system you can unpack list "system" to plot each system, each region is printed here
set = 7
#which picture do you want to plot
cal1, cal2 = region[set]
#number of peaks region
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(xfinaldata[cal1:cal2], yfinaldata[cal1:cal2], "r", label='raw')
ax1.plot(xfinaldata[cal1:cal2], gauss_peak_list[cal1:cal2], "b", label='gaussian fitting')
plt.xlabel("Observed-frame $\lambda$ [$\AA$]")
plt.ylabel("Final Residual")
plt.legend()
plt.show()
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
    sum = (up - down) / 2 * np.dot(wi, (_2gaussian(xi, amp1, cen1, sigma1, amp2, cen2, sigma2)-1))
    return sum
def calcuZ(parameterlist):
    """
    calculate shift of selected MGII system
    should be lower than the shift of qso
    """
    center1 = parameterlist[1]
    center2 = parameterlist[4]
    zmgii = ((center2 - center1)/(2803-2796))-1
    zmgiicorrection1 = (center1/2796)-1
    zmgiicorrection2 = (center2/2803)-1
    return zmgii, zmgiicorrection1, zmgiicorrection2
def calcuequivalent_width(xfinaldata, parameterlist):
    """
    use integration of estimated double gaussian function to
    calculate rest equivalent width of MGII system

    """
    sum = integration(xfinaldata, parameterlist)
    equw = sum/1
    # continuum are already normalized to 1
    return equw
nset = set + 1
zmgii, zmgiicorrection1, zmgiicorrection2 = calcuZ(parameterlist[6*(nset-1):6*nset])
equw = calcuequivalent_width(xfinaldata[cal1:cal2], parameterlist[6*(nset-1):6*nset])
print("equw=", equw)
print("zmgii=", zmgii)
print("zmgiicorrection1=", zmgiicorrection1)
print("zmgiicorrection2=", zmgiicorrection2)
