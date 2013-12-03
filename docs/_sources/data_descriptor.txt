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

Element ReadIteration
=====================

A data descriptor must expose a method element_read_iter_interface(),
returning a subclass of blaze.datadescriptor.IReadElementIter.
This object must be a Python iterator, returning raw pointer
values as Python ints.

It may also expose methods c_read_iter and llvm_read_iter, which return
C function pointers/data and LLVM inlinable code respectively
to do the iteration.

This iteration is for read access only, so if the data is on
disk, compressed, or requires any other form of transformation
the iterator should use a temporary intermediate buffer. Each
iteration step is allowed to invalidate all previous pointers
returned, in order to facilitate such buffering.

The elements pointed to by the pointers returned must be in
C order, contiguous, and follow C alignment of the
platform/compiler Python was built with.

Get Element
===========

A data descriptor must expose a method get_element_interface,
which returns an object which can do a getitem for a fixed
number of integer indices. Its get method must accept
a tuple of nindex integers, and return a raw pointer to
the data.

The returned object may also implement c_getter and llvm_getter,
which produce C function pointers/data and inlinable LLVM
code to do the same get operation.

Any get operation invalidates all previous pointers returned.
This interface requirement is so that a buffer/element cache
may be used by the implementation.

The elements pointed to by the pointers returned must be in
C order, contiguous, and follow C alignment of the
platform/compiler Python was built with.

Element WriteIteration
======================

This is the equivalent of ReadIteration for writing to
a writable data descriptor. Details to be fleshed out.

Set Element
===========

This is the equivalent of get element for writing to
a writable data descriptor. Details to be fleshed out.

