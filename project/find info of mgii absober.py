import os
import numpy as np

import fitsio

from astropy.io import fits
from astropy.table import Table, vstack, join

import desispec.io
info_list = ['main', 'dark']
survey, program = info_list
specprod = "/%s/%s" % (survey, program)  # Internal name for the EDR
specprod_dir = "/home/haixiangshu/Documents/global/cfs/cdirs/desi/spectro/redux/guadalupe/"
#%%
guadalupe = Table(fitsio.read(os.path.join(specprod_dir, "zcatalog", 'zpix-%s-%s.fits' % (survey, program))))
t_fivespec = guadalupe[guadalupe["ZCAT_NSPEC"]==5]

#-- unique TARGETID of each object with five spectra
targids = np.unique(t_fivespec["TARGETID"])

print("\tTARGETID\t\tSPECTYPE of all 5 spectra")
for i,tid in enumerate(targids):
    these_spec = t_fivespec[t_fivespec["TARGETID"]==tid]
    spectype   = these_spec["SPECTYPE"].data.astype(str)
    print("{0:3}\t{1}\t{2}".format(i+1, tid, spectype))
from astropy.table import Table
redrock = Table.read('redrock-6-7951-thru20211014.fits', 'REDSHIFTS')
ii = redrock['Z']>4
redrock['TARGETID', 'SPECTYPE', 'Z'][ii]