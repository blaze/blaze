======================
Release Notes
======================

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

