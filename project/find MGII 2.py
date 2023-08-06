import numpy as np
from pathlib import Path
import functools
from astropy.convolution import convolve
from sklearn import mixture
from astropy.io import fits
from astropy.table import Table, vstack, hstack, unique, join
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pylab import *
import os
# sys.path.append("/global/u2/b/bid13/VI/prospect/py")
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.optimize import curve_fit
from scipy.special import roots_legendre
import desispec.coaddition
import desispec.io
import desispec.spectra
# from desitarget.cmx.cmx_targetmask import cmx_mask
# from desitarget.sv1.sv1_targetmask import desi_mask, bgs_mas
def interp_grid(xval, xarr, yarr):
    """Basic linear interpolation of [`xarr`, `yarr`] on point `xval`.

    Translated from js/interp_grid.js.

    Parameters
    ----------
    xval : :class:`xval`
        Interpolate y-value at this point.
    xarr, yarr : array-like
        X, Y data.

    Returns
    -------
    :class:`float`
        The y-value corresponding to `xval`.
    """
    index = index_dichotomy(xval, xarr)  # return i_left in xarr
    a = (yarr[index + 1] - yarr[index]) / (xarr[index + 1] - xarr[index])
    b = yarr[index] - a * xarr[index]
    yval = a * xval + b
    return yval


# might output close to (yarr[index+1]+yarr[index])/2
def index_dichotomy(point, grid):
    """Find nearest index in `grid`, left from `point`; use dichotomy method.

    Translated from js/interp_grid.js.

    Parameters
    ----------
    point : :class:`float`
        Value to find in `grid`.
    grid : array-like
        Values to search.

    Returns
    -------
    :class:`int`
        Nearest index.
    """
    if point < grid[0]:
        return 0
    if point > grid[-1]:
        return len(grid) - 2
    i_left = 0
    i_center = 0
    i_right = len(grid) - 1
    while i_right - i_left != 1:
        i_center = i_left + np.floor((i_right - i_left) / 2)
        i_center = int(i_center)
        if point >= grid[i_center]:
            i_left = i_center
        else:
            i_right = i_center
    return i_left


# dichotomy to find the nearest element of point in grid only if element are in sequence
def coadd_brz_cameras(wave_in, flux_in, noise_in, mask_in):
    """Camera-coadd *brz* spectra.

    Translated from js/coadd_brz_cameras.js.

    Parameters
    ----------
    wave_in : array-like
        Set of three wavelength arrays corresponding to *brz*.
    flux_in : array-like
        Set of three flux arrays corresponding to *brz*.
    noise_in : array-like
        Noise arrays for weighting.

    Returns
    -------
    :func:`tuple`
        The coadded wavelength solution, flux and noise.

    Notes
    -----
    * Need to handle case of no noise.
    """

    # Find b,r,z ordering in input arrays
    wave_start = [wave_in[0][0], wave_in[1][0], wave_in[2][0]]
    i_b = wave_start.index(np.amin(wave_start))
    i_z = wave_start.index(np.amax(wave_start))  # out put number between (0:2)
    i_r = 1
    for i in [0, 1, 2]:
        if ((i_b != i) and (i_z != i)): i_r = i  # output the third index except from i_b and i_z

    wave_out = []
    flux_out = []
    noise_out = []
    mask_out = []
    margin = 20
    # wave_out is a 3 x n array
    for i in range(len(wave_in[i_b])):  # length in row of wave_in
        if (wave_in[i_b][i] < wave_in[i_b][-1] - margin):
            wave_out.append(wave_in[i_b][i])
            flux_out.append(flux_in[i_b][i])
            noise_out.append(noise_in[i_b][i])
            mask_out.append(mask_in[i_b][i])
    the_lim = wave_out[-1]  # append elements< last element -20
    for i in range(len(wave_in[i_r])):  # r
        if ((wave_in[i_r][i] < wave_in[i_r][-1] - margin) and (wave_in[i_r][i] > the_lim)):
            wave_out.append(wave_in[i_r][i])
            flux_out.append(flux_in[i_r][i])
            noise_out.append(noise_in[i_r][i])
            mask_out.append(mask_in[i_r][i])
    the_lim = wave_out[-1]  # append elements in wave_out[-1] <x< the last element of wave_in[i_r][-1]-20
    for i in range(len(wave_in[i_z])):  # z
        if (wave_in[i_z][i] > the_lim):
            wave_out.append(wave_in[i_z][i])
            flux_out.append(flux_in[i_z][i])
            noise_out.append(noise_in[i_z][i])
            mask_out.append(mask_in[i_z][i])  # maybe make elements of wave_in in sequence
            # all_out are 1 dimention array
    for i in range(len(wave_out)):  # combine in overlapping regions
        b1 = -1
        b2 = -1
        if ((wave_out[i] > wave_in[i_r][0]) and (wave_out[i] < wave_in[i_b][-1])):  # br
            b1 = 0
            b2 = 1  # comfirm if elements are overlapping
        if ((wave_out[i] > wave_in[i_z][0]) and (wave_out[i] < wave_in[i_r][-1])):  # rz
            b1 = 1
            b2 = 2  # comfirm if elements are overlapping
            # the code upstairs would miss some elements
        if (b1 != -1):  # overlapping do happens
            phi1 = interp_grid(wave_out[i], wave_in[b1], flux_in[b1])  # function that combine first2 interlace spectra
            noise1 = interp_grid(wave_out[i], wave_in[b1], noise_in[b1])
            phi2 = interp_grid(wave_out[i], wave_in[b2], flux_in[b2])
            noise2 = interp_grid(wave_out[i], wave_in[b2],
                                 noise_in[b2])  # function that combine later2 interlace spectra
            mask1 = interp_grid(wave_out[i], wave_in[b1], mask_in[b1])
            mask2 = interp_grid(wave_out[i], wave_in[b2], mask_in[b2])
            if mask1 + mask2:
                mask_out[i] = 1
            if (noise1 > 0 and noise2 > 0):
                iv1 = 1 / (noise1 * noise1)
                iv2 = 1 / (noise2 * noise2)
                iv = iv1 + iv2
                noise_out[i] = 1 / np.sqrt(iv)  # n1*n2/sqrt(n1**2+n2**2)
                flux_out[i] = (iv1 * phi1 + iv2 * phi2) / iv  # blablabla
    return (np.asarray(wave_out), np.asarray(flux_out), np.asarray(noise_out),
            np.asarray(mask_out))  # turn into array for each i
info_list = [39637324216140592, 'guadalupe', 'main', 'dark', 9738]
# %%
targ_id, release, survey, program, pix = info_list
path = '/home/haixiangshu/Documents/global/cfs/cdirs/desi/spectro/redux/%s/healpix/%s/%s' % (release, survey, program)
healpix_loc = path + '/' + str(pix)[0:2] + '/' + str(pix)
spectra = healpix_loc + '/coadd-%s-%s-%s.fits' % (survey, program, pix)
h_coadd = fits.open(spectra)
h_coadd.info()
cat_path = '/home/haixiangshu/Documents/global/cfs/cdirs/desi/spectro/redux/guadalupe/zcatalog'
cat_name = 'zpix-%s-%s.fits' % (survey, program)
cat_path = os.path.join(cat_path, cat_name)
zcat = Table.read(cat_path, hdu=1)
#print(zcat[50000:50050])
#some of this target are in other files
#get accessible target ID
zcat.add_index('TARGETID')
z = zcat.loc[targ_id]['Z']
#get accessible target ID
sp_petal = Table.read(spectra, hdu="FIBERMAP")[100:140]
print(sp_petal)
print("z=", z)
with fits.open(spectra) as spec:
    ids = spec['FIBERMAP'].data['TARGETID']
    targ_idx = targ_id == ids
    wave_b = spec['B_WAVELENGTH'].data.copy()
    wave_r = spec['R_WAVELENGTH'].data.copy()
    wave_z = spec['Z_WAVELENGTH'].data.copy()
    wave_in = [wave_b, wave_r, wave_z]
    # 3 spectra in the helpix
    flux_b = spec['B_FLUX'].data[targ_idx, :][0].copy()
    flux_r = spec['R_FLUX'].data[targ_idx, :][0].copy()
    flux_z = spec['Z_FLUX'].data[targ_idx, :][0].copy()
    flux_in = [flux_b, flux_r, flux_z]

    ivar_b = spec['B_IVAR'].data[targ_idx, :][0].copy()
    ivar_r = spec['R_IVAR'].data[targ_idx, :][0].copy()
    ivar_z = spec['Z_IVAR'].data[targ_idx, :][0].copy()
    noise_in = [1 / ivar_b ** 0.5, 1 / ivar_r ** 0.5, 1 / ivar_z ** 0.5]

    mask_b = spec['B_MASK'].data[targ_idx, :][0]
    mask_r = spec['R_MASK'].data[targ_idx, :][0]
    mask_z = spec['Z_MASK'].data[targ_idx, :][0]
    mask_in = [mask_b, mask_r, mask_z]

wave, flux, noise, mask = coadd_brz_cameras(wave_in, flux_in, noise_in, mask_in)


def normalize_initial(flux, invar, wave, z):
    """
        param flux;inver;mask;wavelength;w_low,w_high for normalization range;

        return:normalized flux,inverse vairiance, mask map and wavelength

    """
    nan_idx = np.isnan(flux)  # choose numbers from texts
    flux[nan_idx] = 0
    invar[nan_idx] = 0
    # return texts to zero

    del_wave = 0.8 / 5600 / np.log(10)  # q0 6.204206884332168e-05 dwave in log space
    wave_ma = np.arange(np.log10(500), np.log10(10000), del_wave)  # q1
    # a wider wave in log space
    dens_fix_ma = (10 ** (wave_ma + (del_wave / 2)) - 10 ** (wave_ma - (del_wave / 2)))
    # 10**wave_ma*(10**(del_wave / 2)-10**(- (del_wave / 2)))
    logwave = np.log10(wave)
    dens_fix = (10 ** (logwave + (del_wave / 2)) - 10 ** (logwave - (del_wave / 2)))

    rest_wave = wave / (1 + z)  # wave in the rest frame
    norm = np.mean(flux)
    flux = flux / norm
    invar = invar * (norm ** 2)
    corflux = flux * dens_fix

    corivar = invar / (dens_fix ** 2)

    corwave = np.log10(rest_wave)
    # coordinate in log10 space
    base_shift = np.modf((corwave[0] - wave_ma[0]) / del_wave)  # q3
    # shift in log space
    s_shift = base_shift[1]
    # the int part of shift
    fmodL = base_shift[0]
    # the decimals part of shift
    fmodR = 1 - fmodL
    beta = ((fmodL * np.concatenate(([0], corflux))) + (
                fmodR * np.concatenate((corflux, [0]))))  # the concatenate use to smooth
    alpha = ((fmodL * np.concatenate(([0], corivar))) + (
                fmodR * np.concatenate((corivar, [0]))))  # the concatenate use to smooth
    endfix = np.int64(wave_ma.shape[0] - np.int64(s_shift) - beta.shape[0])  #
    mf = np.concatenate((np.zeros(np.int64(s_shift)), beta, np.zeros(endfix)))  #
    mi = np.concatenate((np.zeros(np.int64(s_shift)), alpha, np.zeros(endfix)))  #
    mf = mf / dens_fix_ma
    mi = mi * (dens_fix_ma ** 2)
    loc_cut = np.where((10 ** wave_ma > 0) & (10 ** wave_ma < 15000))
    # select wave between maxwave and minmum wave (search window)
    flux_new, invar_new, wave_ma = np.array(mf[loc_cut]), np.array(mi[loc_cut]), wave_ma[loc_cut]
    return flux_new, invar_new, wave_ma
flux1, noise1, wave1=normalize_initial(flux, noise, wave, z)


wave_cont = 10 ** wave1 * (1 + z)



plt.figure(figsize = (20, 6))


plt.plot(10 ** wave1 * (1 + z), flux1, color = 'r', alpha = 0.5)
# in the observed frame
# Over-plotting smoothed spectra in black for all the three arms
plt.plot(wave_cont, convolve(flux1, Gaussian1DKernel(5)), color = 'k')

plt.xlim([3500, 10000])
plt.ylim([0, 5])
plt.xlabel('$\lambda$ [$\AA$]')
plt.ylabel('Flux (Arbitrary Unit)')

plt.figure(figsize = (20, 6))

convolve = convolve(flux1, Gaussian1DKernel(5))
convolve = np.where(convolve == np.inf, np.nan, convolve)
convolve = np.where(convolve == -np.inf, np.nan, convolve)
normalized_flux = flux1/convolve
# Over-plotting smoothed spectra in black for all the three arms
plt.plot(wave_cont, normalized_flux, color = 'k')
# in the observed frame
plt.xlim([3500, 10000])
plt.ylim([0, 2])
plt.xlabel('$\lambda$ [$\AA$]')
plt.ylabel('NMF Residual')

#plt.show()

def Criterion_MGII(flux, noise, continuum_flux, wavelength, z):
    """
            param flux;noise;continuum_flux;wavelength:
            use method:
            SNR(MGII lamda2796)>4 && SNR(MGII lamda2803)>2
            return:location in shift(0~3)
    """
    location1 = np.zeros((len(flux)))
    location2 = np.zeros((len(flux)))
    noise = noise * (continuum_flux) ** 2
    noise = 1 / noise ** 0.5
    wavemax = 2803 * (1+z)
    observe_wave = 10 ** wavelength * (1 + z)
    for ii in range(len(flux)):
        signal = np.absolute((flux[ii]/continuum_flux[ii])-1)
        SNR = signal/noise[ii]
        location1[ii] = np.where(SNR > 2 and wavemax > observe_wave[ii], observe_wave[ii], 0)
        location2[ii] = np.where(SNR > 4 and wavemax > observe_wave[ii], observe_wave[ii], 0)
        # SNR number is replaceable, to search for best fitting you need to select by your self
        # the mgii system are observed in front of qso
    return location1, location2
location1, location2 = Criterion_MGII(flux1, noise1, convolve, wave1, z)

def split_candidates(final_loc, final_locrow, final_loc1, final_locrow1, z, system):
    """
    the specific length of each candidate to split array into specific length subarray
    """

    waverangemax = (2803 - 2796) * (1 + z)
    # we decide range od each mgii peak is less than 5
    for k in range(len(final_loc)):
        if k > 0:
            if np.max(final_locrow) - (np.min(final_locrow) + 5) <= waverangemax:
                final_loc1.append(final_loc[0:len(final_loc)-1])
                final_locrow1.append(final_locrow[0:len(final_loc)-1])
                system.append(len(final_loc)-1)
                return system
            if final_locrow[k] - (np.min(final_locrow) + 5) > waverangemax:
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

def parameter(location1, location2, flux, wavelength, z):
    """
                param flux;locmgii1;locmgii2;wavelength,z:
                use method:
                double Gaussian profile to fit doublet candidates
                return:location of final MGII , shift of each system
    """
    loc = []
    locrow = []
    wavelength = 10 ** wavelength * (1 + z)
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
    system, final_loc1, final_locrow1, d = split_candidates(final_loc, final_locrow, final_loc1, final_locrow1, z, system)
    system = np.array(system)
    m1 = []
    m2 = []
    w1 = []
    w2 = []
    c1 = []
    c2 = []
    num1 = 0
    cal = 0
    #each k repesent one possible mgii system
    for k in range(len(system)-1):
        cal = num1
        num1 += system[k]
        xdata = []
        ydata = []
        #do double gaussian fix
        for i in range(cal, num1):
            locval = final_loc1[i]
            xdata.append(wavelength[locval])
            ydata.append(flux[locval])
        gaussian = mixture.GaussianMixture(n_components=2, covariance_type='full')
        gaussian.fit((ydata, xdata))
        m11, m22 = gaussian.means_
        w11, w22 = gaussian.weights_
        c11, c22 = gaussian.covariances_
        m1.append(m11), m2.append(m22), w1.append(w11), w2.append(w22), c1.append(c11), c2.append(c22)
    return m1, m2, w1, w2, c1, c2
m1, m2, w1, w2, c1, c2 = parameter(location1, location2, flux1, wave1, z)


# Define the Gaussian function

def _1gaussian(x_array, amp1,cen1,sigma1):
    """
        single gaussian function
    """
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2)))
def _2gaussian(x_array, amp1, cen1, sigma1, amp2, cen2, sigma2):
    """
        double gaussian function
    """
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2))) + \
            amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen2)/sigma2)**2)))
def suit_system(location1, location2, wavelength, z):
    """
                param flux;location1;location2;wavelength,z:
                use method:
                double Gaussian profile to fit doublet candidates
                return:location of final MGII , shift of each system
    """
    loc = []
    locrow = []
    wavelength = 10 ** wavelength * (1 + z)
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
    system, final_loc1, final_locrow1, d = split_candidates(final_loc, final_locrow, final_loc1, final_locrow1, z, system)
    system = np.array(system)
    return system, final_loc1, final_locrow1

def double_Gaussian_profile(location1, location2, flux, wavelength, z):
    #each n repesent one possible mgii system
    system, final_loc1, final_locrow1 = suit_system(location1, location2, wavelength, z)
    wavelength = 10 ** wavelength * (1 + z)
    num1 = 0
    cal = 0
    xfinaldata = []
    yfinaldata = []
    parameterlist = []
    coveriancelist = []
    xdata = []
    ydata = []
    gauss_peak_list = []
    for n in range(len(system)-1):
        cal = num1
        num1 += system[n]
        #do double gaussian fix
        for i in range(cal, num1):
            locval = final_loc1[i]
            xdata.append(wavelength[locval])
            ydata.append(flux[locval])
        x_array = np.asarray(xdata)
        y_array_2gauss = np.asarray(ydata)
        nu = len(x_array[cal: num1])
        center = sum(x_array[cal: num1]) / nu
        mean = sum(x_array[cal: num1]*y_array_2gauss[cal: num1]) / nu
        sigma = sum(y_array_2gauss[cal: num1] * (x_array[cal: num1] - mean) ** 2) / nu
        # Define the Gaussian function
        # we estimate parameter below for reducing calculate
        amp1 = 1
        cen1 = center
        sigma1 = sigma
        amp2 = 1
        cen2 = center+(2803 - 2796)
        sigma2 = sigma
        _2gauss = _2gaussian(x_array[cal: num1], amp1, cen1, sigma1, amp2, cen2, sigma2)
        popt_2gauss, pcov_2gauss = curve_fit(_2gaussian, x_array[cal: num1], y_array_2gauss[cal: num1], maxfev = 80000000, p0=[amp1, cen1, sigma1, amp2, cen2, sigma2])
        pcov_2gauss = np.sqrt(np.diag(pcov_2gauss))
        gauss_peak_main = _2gaussian(x_array[cal: num1], *popt_2gauss)
        gauss_peak_main = list(gauss_peak_main)
        popt_2gauss = list(popt_2gauss)
        pcov_2gauss = list(pcov_2gauss)
        gauss_peak_list = gauss_peak_list + gauss_peak_main
        parameterlist = parameterlist +popt_2gauss
        coveriancelist = coveriancelist + pcov_2gauss
    return x_array, y_array_2gauss, parameterlist, coveriancelist, gauss_peak_list, system


x_array, y_array_2gauss, parameterlist, coveriancelist, gauss_peak_list, system = double_Gaussian_profile(location1, location2, normalized_flux, wave1, z)
region = []
num1 = 0
cal = 0
for o in range(len(system)-1):
    cal = num1
    num1 += system[o]
    region.append([cal, num1])
print(region)
#for every system you can unpack list "system" to plot each system, each region is printed here

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x_array[62: 78], y_array_2gauss[62: 78], "r", label='raw')
ax1.plot(x_array[62: 78], gauss_peak_list[62: 78], "b", label='gaussian fitting')
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
def calcuequivalent_width(x_array, parameterlist):
    """
    use integration of estimated double gaussian function to
    calculate rest equivalent width of MGII system

    """
    sum = integration(x_array, parameterlist)
    equw = sum/1
    # continuum are already normalized to 1
    return equw
zmgii, zmgiicorrection1, zmgiicorrection2 = calcuZ(parameterlist[6*2:6*3])
equw = calcuequivalent_width(x_array[62:78], parameterlist[6*2:6*3])
print("equw=", equw)
print("zmgii=", zmgii)
print("zmgiicorrection1=", zmgiicorrection1)
print("zmgiicorrection2=", zmgiicorrection2)