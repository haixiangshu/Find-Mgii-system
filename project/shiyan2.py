import numpy as np
from pathlib import Path
import functools
from astropy.convolution import convolve
from astropy.io import fits
from astropy.table import Table, vstack, hstack, unique,join
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from pylab import *
# sys.path.append("/global/u2/b/bid13/VI/prospect/py")
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.optimize import curve_fit
import desispec.coaddition
import desispec.io
import desispec.spectra
# from desitarget.cmx.cmx_targetmask import cmx_mask
#from desitarget.sv1.sv1_targetmask import desi_mask, bgs_mas
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
info_list = [39632941462061599, 'guadalupe', 'main', 'dark', 9738]
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

convolve = convolve(flux1, Gaussian1DKernel(5))
convolve = np.where(convolve == np.inf, np.nan, convolve)
convolve = np.where(convolve == -np.inf, np.nan, convolve)
normalized_flux = flux1/convolve
# Over-plotting smoothed spectra in black for all the three arms
plt.plot(wave_cont, normalized_flux, color = 'k')
# in the observed frame
plt.xlim([3550, 7000])
plt.ylim([0, 2])
plt.xlabel('$\lambda$ [$\AA$]')
plt.ylabel('$F_{\lambda}$ [$10^{-17} erg\ s^{-1}\ cm^{-2}\ \AA^{-1}$]')

plt.figure(figsize = (20, 6))

plt.show()