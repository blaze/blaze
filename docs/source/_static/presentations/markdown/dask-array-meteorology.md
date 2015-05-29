## Meteorological data

We have a pile of NetCDF files

    $ ls
    2014-01-01.nc  2014-03-18.nc  2014-06-02.nc  2014-08-17.nc  2014-11-01.nc
    2014-01-02.nc  2014-03-19.nc  2014-06-03.nc  2014-08-18.nc  2014-11-02.nc
    2014-01-03.nc  2014-03-20.nc  2014-06-04.nc  2014-08-19.nc  2014-11-03.nc
    2014-01-04.nc  2014-03-21.nc  2014-06-05.nc  2014-08-20.nc  2014-11-04.nc
    ...             ...             ...             ...             ...

Four measurements per day, quarter degree resolution, for 2014

    >>> import netCDF4
    >>> t = netCDF4.Dataset('2014-01-01.nc').variables['t2m']
    >>> t.shape
    (4, 721, 1440)


## Meteorological data

Point to a bunch of NetCDF datasets

    >>> filenames = sorted(glob('2014-*.nc'))
    >>> temps = [netCDF4.Dataset(fn).variables['t2m'] for fn in filenames]

Wrap each with `dask.array`

    >>> import dask.array as da
    >>> arrays = [da.from_array(t, chunks=(4, 200, 200)) for t in temps]

Manipulate arrays with numpy syntax

    >>> x = da.concatenate(arrays, axis=0)
    >>> x.shape
    (1464, 721, 1440)


## Meteorological data

Interact with the ecosystem

    >>> from matplotlib import imshow
    >>> imshow(x.mean(axis=0), cmap='bone')

<img src="images/avg.png" width="100%">


## Meteorological data

Interact with the ecosystem

    >>> from matplotlib import imshow
    >>> imshow(x[1000] - x.mean(axis=0), cmap='RdBu_r')

<img src="images/diff.png" width="100%">


## Meteorological data

Interact with the ecosystem

    >>> from matplotlib import imshow
    >>> imshow(x[::4].mean(axis=0) - x[2::4].mean(axis=0), cmap='RdBu_r')

<img src="images/day-vs-night.png" width="100%">



## XRay

<img src="images/xray-logo.png" width="30%">

Dask.array integrates with XRay.
[http://xray.readthedocs.org](http://xray.readthedocs.org)

<hr>

*  Implements the netCDF model
    *  Set of associated ndarrays / variables
    *  Pandas index along each axis
*  Index and reason using named axes with labels
    * NumPy -- `x[40:100].mean(axis=2)`
    * XRay -- `ds.sel(time='2014-04').mean('time')`

<hr>

Written by Stephan Hoyer (@shoyer) at Climate Corp



