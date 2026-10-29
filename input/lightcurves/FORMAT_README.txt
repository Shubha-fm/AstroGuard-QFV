One CSV per real ALeRCE object. Required columns:
mjd,magpsf,sigmapsf,fid
Example: fid=1=g, fid=2=r. Each light curve needs >=20 valid measurements.
The object manifest must refer to its file, relative to input/objects.csv, e.g. lightcurves/ZTFxxxx.csv
Real ALeRCE data are fetched only by collect stage or provided separately. This package contains NO simulated astrophysical data.
