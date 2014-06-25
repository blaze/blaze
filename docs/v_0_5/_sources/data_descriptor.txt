================
Data Descriptors
================

Data Descriptors are unadorned blaze arrays, objects which
expose data with a dshape but have no math or other facilities
attached. The data descriptor interface is analogous to the
Python buffer interface described in PEP 3118, but with some
more flexibility.

An object signals it is a data descriptor by subclassing from
the blaze.datadescriptor.DataDescriptor abstract base class,
and implementing the necessary methods.

Python-Style Iteration
======================

A data descriptor must expose the Python __iter__ method,
which should iterate over the outermost dimension of the data.
If the data is a scalar, it should raise an IndexError.

This iterator must return only data descriptors for the
iteration, and if the data is writable, the returned data descriptors
must point into the same data, not contain copies. This can
be tricky in some cases, for example NumPy returns a copy if
it is returning a scalar.

This style of iteration can handle variable-sized/ragged arrays,
because access is handled in a nested fashion one dimension at
a time. Element iteration cannot handle this kind of dimension,
because it requires that the data conform to a C storage layout
that does not include variable-sized arrays presently.

Note that this style of iteration cannot be used to get at
actual data values. For this, you must use the element
access mechanisms.

Python-Style GetItem
====================

A data descriptor must expose the Python __getitem__ method,
which should at a minimum support tuples of integers as
input.

Calling the [] operator on a data descriptor must
return another data descriptor. Just as for the iterator,
when the data is writable the returned data descriptors
must point into the same data.

Note that this style of access cannot be used to get at
actual data values. For this, you must use the element
access mechanisms.

Memory Access
=============

To get data from the data descriptors in a raw memory form,
there is a way to retrieve it as a DyND array via the
``ddesc.dynd_arr()`` method
