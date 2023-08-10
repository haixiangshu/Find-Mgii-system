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


def coupling_flux_and_fluxerr_arrays():
    other_flux = []
    other_fluxerr = []
    matched_fluxerr = []

    # First, find all arrays that have to do with flux (using TUTYP)
    for i in other_arrays:  # i is the FITS index (starts from 1, not from 0)
        if 'data.fluxaxis.value' in utype[i - 1]:
            other_flux.append(i)
        if 'data.fluxaxis.accuracy.staterror' in utype[i - 1]:
            other_fluxerr.append(i)

    print("Coupling flux and fluxerror arrays:")
    print("    - Coupled:           %s - %s" % (name[iflux - 1], name[ierr - 1]))

    for i in other_flux:
        i_namespace_length = utype[i - 1].find(':')
        i_namespace = utype[i - 1][:i_namespace_length]
        matched_couple = 0
        # for this flux, seek the corresponding fluxerr
        for j in other_fluxerr:
            j_namespace_length = utype[j - 1].find(':')
            j_namespace = utype[j - 1][:j_namespace_length]
            if i_namespace == j_namespace:
                print("    - Coupled:           %s - %s - namespace: %s " % (name[i - 1], name[j - 1], i_namespace))
                matched_couple = 1
                matched_fluxerr.append(j)
        if matched_couple == 0:
            # no matching fluxerr could be found, reporting:
            print("    - Uncoupled flux:    %s - utype: %s" % (name[i - 1], Utype[i - 1]))

    # Above the unmatched flux arrays were reported;
    # here we report the unmatched fluxerr arrays (if any).

    unmatched_fluxerr = list(set(other_fluxerr) - set(matched_fluxerr))
    for i in unmatched_fluxerr:
        print("    - Uncoupled fluxerr: %s - utype: %s" % (name[i - 1], Utype[i - 1]))

    print("")


# *********************************************
#                    MAIN
# *********************************************


cat_path = '/home/haixiangshu/Documents/UVES'
cat_name = 'ADP.2019-03-12T05_26_45.026.fits'
cat_path = os.path.join(cat_path, cat_name)
zcat = Table.read(cat_path, hdu=1)

hdulist = pyfits.open(cat_path)   # Open FITS file

phu = hdulist[0].header        # Primary Header Unit: metadata
scihead = hdulist[1].header       # Header of the first FITS extension: metadata
scidata = hdulist[1].data         # Data in the first FITS extension: the spectrum

# Checking some compliance
# 1. keyword PRODCATG must be present in primary header unit
try:
    prodcatg = phu['PRODCATG']
except:
    errorMessage = 'Keyword PRODCATG not found in primary header.\nFile not compliant with the ESO Science Data Product standard.'
    print(errorMessage)
    exit(1)

# 2. value of keyword PRODCATG must match SCIENCE.SPECTRUM*
if not prodcatg.startswith('SCIENCE.SPECTRUM'):
    errorMessage = "Expected header keyword: PRODCATG = 'SCIENCE.SPECTRUM'\nFound: PRODCATG = '%s'\nFile not compliant with the 1d spectrum specifications\nof the ESO Science Data Product standard." % prodcatg
    print(errorMessage)
    exit(1)

# 3. Various keywords must be defined, among them the ones here below:
try:
    origfile=phu['ORIGFILE']    # Original filename as assigned by the data provider
    instrume = phu['INSTRUME']  # Name of the instrument
    wavelmin = phu['WAVELMIN']  # Minimum wavelength in nm
    wavelmax = phu['WAVELMAX']  # Maximum wavelength in nm
    respower = phu['SPEC_RES']  # Spectral resolving power (lambda / delta_lambda)
    snr      = phu['SNR']       # Signal to Noise Ratio
    specaxisucd  = scihead['TUCD1'] # Gives the type of spectral axis (see SPECTRAL AXIS below)
except:
   errorMessage='File not compliant with the 1D spectrum specifications of the ESO Science Data Product standard; some of the mandatory keywords were not found in primary header unit'
   print('ERROR = %s' % (errorMessage))
   exit(1)

# SPECTRAL AXIS: could be either wavelength, frequency, or energy;
# if wavelength, the distinction between wavelength in air or in vacuum is provided by the presence of the obs.atmos token in the TUCD1.
# the variable spectype will carry to whole info.
spectype = None
if specaxisucd.startswith('em.wl'):
    if specaxisucd == 'em.wl':
        spectype = 'wavelength in vacuum (TUCD1=%s)' % specaxisucd
    elif specaxisucd == 'em.wl;obs.atmos':
        spectype = 'wavelength in air (TUCD1=%s)' % specaxisucd
    else:
        spectype = 'wavelength (TUCD1=%s)' % specaxisucd
elif specaxisucd.startswith('em.freq'):
    spectype = 'frequency (TUCD1=%s)' % specaxisucd
elif specaxisucd.startswith('em.ener'):
    spectype = 'energy (TUCD1=%s)' % specaxisucd

# Report main characteristics of the spectrum:
#print('************************************************************************************************************************')
print('Instrume=%s   Wmin=%snm   Wmax=%snm   R=%s   SNR=%s'  % (instrume,wavelmin,wavelmax,respower,snr))
print('Spectral axis: %s' % (spectype))
print('------------------------------------------------------------------------------------------------------------------------')
try:
    voclass=scihead['VOCLASS']
except:
    exit(1)

# TFIELDS is a required FITS binary table keyword
try:
    tfields = int(scihead['TFIELDS'])
except:
    exit(1)

#################################
# METADATA PART
#################################

# Reading name, unit, utype for each column (array) in the FITS binary table (extension 1).

name = []
unit = []
utype= [] # lowercase utype string: for case-insensitive matches
Utype= [] # original utype, with case preserved: for display

print("AVAILABLE ARRAYS:")
print ("name            index  UNIT                               UTYPE")
for i in range(1, tfields+1):
    thisname = scihead['TTYPE'+str(i)]
    try:
       thisunit = scihead['TUNIT'+str(i)]
    except:
       thisunit=""
    try:
       thisutype=scihead['TUTYP'+str(i)]
    except:
       thisutype='no_utype_assigned:field_not_part_of_the_standard'
    print ("%-15s %2d     %-34s [%-s]" % (thisname, i, thisunit, thisutype))
    name.append(thisname)
    unit.append(thisunit)
    utype.append(thisutype.lower())
    Utype.append(thisutype)

print('------------------------------------------------------------------------------------------------------------------------')

# Recognising the main scientific arrays (spectral, flux and flux error) and the "other" ones.
# A 1D spectrum can contain several flux (and fluxerror) arrays, but one is defined to be the best.
# The best one can be recognised by the (lowercased) utype which is either "spectrum.data.fluxaxis.value" or "spec:data.fluxaxis.value".

other_arrays = []  # It will contain the indeces of the fields not considered main arrays. FITS indeces starts from 1!

# Getting the indexes of the FITS columns
# for the main spectral array (ispec), flux array (iflux), and flux_error (ierr) array:
ispec = 0
iflux = 0
ierr = 0
for i in range(1, tfields+1):

     # Remember that the index of Python arrays starts from 0, while the FITS index from 1.
     tutyp=utype[i-1]

     # The ESO Science Data Product standard format
     # prescribes the spectral axis to be stored in column 1;
     # there would be no need to look for it, but we need the other_arrays anyway.

     # The TUTYPn keywords follow either the Spectrum Data Model standard v1.1 for spectra with a single flux array,
     # or the Spectral Data Model standard v2.0 for spectra with any number of flux arrays
     # These data model standards are available from the International Virtual Observatory Alliance
     # web site at: http://ivoa.net/documents/

     if tutyp == 'spectrum.data.spectralaxis.value':
         ispec = i
     elif tutyp == 'spec:data.spectralaxis.value':
         ispec = i
     elif tutyp == 'spectrum.data.fluxaxis.value':
         iflux = i
     elif tutyp == 'spec:data.fluxaxis.value':
         iflux = i
     elif tutyp == 'spectrum.data.fluxaxis.accuracy.staterror':
         ierr  = i
     elif tutyp == 'spec:data.fluxaxis.accuracy.staterror':
         ierr  = i
     else:
         # Storing the indeces of other, not considered main, arrays:
         other_arrays.append( i )

coupling_flux_and_fluxerr_arrays()

# --------------------------------------------------------------------------------------------------------


# Number of points in the spectrum: NELEM
NELEM = scihead['NELEM']

print("MAIN ARRAYS:")
print("                   name          index   comment")
print("  Spectral column: %-12s %2d       %s" % (name[ispec - 1], ispec, spectype))
print("      Flux column: %-12s %2d" % (name[iflux - 1], iflux))
print("Flux Error column: %-12s %2d" % (name[ierr - 1], ierr))
print(
    '------------------------------------------------------------------------------------------------------------------------')

#################################
# DATA PART and plots
#################################

print("\nThe spectrum has %d points\n" % (NELEM))

# Main arrays:
spec = np.array(scidata[0][ispec - 1])
flux = np.array(scidata[0][iflux - 1])
err = np.array(scidata[0][ierr - 1])

# To plot using maptplotlib:
import matplotlib as mpl
import matplotlib.pyplot as plt

xlabel_1 = ("%s [%s]" % (name[ispec - 1], unit[ispec - 1]))
ylabel_1 = ("%s [%s]" % (name[iflux - 1], unit[iflux - 1]))

print("Fig 1 will show x=%s vs y=%s (i.e. arrays: %d vs %d)" % (name[ispec - 1], name[iflux - 1], ispec, iflux))
print("Fig 2 will show x=%s vs y= an array of your choice; " % (name[ispec - 1]))

plt.figure()
plt.plot(spec, flux)
#plt.plot(spec, err)
#plt.errorbar(spec, flux, xerr=0, yerr=err)

plt.xlabel( xlabel_1 )
plt.ylabel( ylabel_1 )
plt.show()