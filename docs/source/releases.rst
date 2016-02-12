=============
Release Notes
=============

.. include:: whatsnew/0.9.0.txt

.. include:: whatsnew/0.8.3.txt

.. include:: whatsnew/0.8.2.txt

.. include:: whatsnew/0.8.1.txt

Release 0.7.3
-------------

* General maturation of many backends through use.
* Renamed ``into`` to ``odo``


Release 0.7.0
-------------

* Pull out data migration utilities to ``into`` project
* Out-of-core CSV support now depends on chunked pandas computation
* h5py and bcolz backends support multi-threading/processing
* Remove ``data`` directory including ``SQL``, ``HDF5`` objects.  Depend on
  standard types within other projects instead (e.g. ``sqlalchemy.Table``,
  ``h5py.Dataset``, ...)
* Better support SQL nested queries for complex queries
* Support databases, h5py files, servers as first class datasets


Release 0.6.6
-------------

* Not intended for public use, mostly for internal build systems
* Bugfix

Release 0.6.5
-------------

* Improve uri string handling #715
* Various bug fixes #715

Release 0.6.4
-------------

* Back CSV with ``pandas.read_csv``.  Better performance and more robust
  unicode support but less robust missing value support (some regressions) #597
* Much improved SQL support #626 #650 #652 #662
* Server supports remote execution of computations, not just indexing #631
* Better PyTables and datetime support #608 #639
* Support SparkSQL #592


Release 0.6.3
-------------

* by takes only two arguments, the grouper and apply
  child is inferred using common_subexpression
* Better handling of pandas Series object
* Better printing of empty results in interactive mode
* Regex dispatched resource function bound to Table, e.g.
   ``Table('/path/to/file.csv')``


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

