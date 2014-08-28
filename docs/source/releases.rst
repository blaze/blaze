======================
Release Notes
======================

Release 0.6.2
-------------

* Efficient CSV to SQL migration using native tools #454
* Dispatched ``drop`` and ``create_index`` functions  #495
* DPlyr interface at ``blaze.api.dplyr``.  #484
* Various bits borrowed from that interface
    * ``transform`` function adopted to main namespace
    * ``Summary`` object for named reductions
    * Keyword syntax in ``by`` and ``merge`` e.g.
      ``by(t, t.col, label=t.col2.max(), label2=t.col2.min())``
* New Computation Server  #527
* Better PyTables support  #487 #496 #526


Release 0.6.1
-------------

* More consistent behavior of ``into``
* ``bcolz`` backend
* Control namespace leakage

Release 0.6
-----------

* Nearly complete rewrite
* Add abstract table expression system
* Translate expressions onto a variety of backends
* Support Python, NumPy, Pandas, h5py, sqlalchemy,
  pyspark, PyTables, pymongo

Release 0.5
-----------

* HDF5 in catalog.
* Reductions like any, all, sum, product, min, max.
* Datetime design and some initial functionality.
* Change how Storage and ddesc works.
* Some preliminary rolling window code.
* Python 3.4 now in the test harness.

Release 0.4.2
-------------

* Fix bug for compatibility with numba 0.12
* Add sql formats
* Add hdf5 formats
* Add support for numpy ufunc operators

Release 0.4.1
-------------

* Fix bug with compatibility for numba 0.12

Release 0.4
-----------

* Split the datashape and blz modules out.
* Add catalog and server for blaze arrays.
* Add remote arrays.
* Add csv and json persistence formats.
* Add python3 support
* Add scidb interface

Release 0.3
-----------

* Solidifies the execution subsystem around an IR based
  on the pykit project, as well as a ckernel abstraction
  at the ABI level.
* Supports ufuncs running on ragged array data.
* Cleans out previous low level data descriptor code,
  the data descriptor will have a higher level focus.
* Example out of core groupby operation using BLZ.

Release 0.2
-----------

* Brings in dynd as a required dependency
  for in-memory data.

Release 0.1
-----------

* Initial preview release

