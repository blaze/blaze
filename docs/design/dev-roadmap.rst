===================
Development Roadmap
===================

This document sets out a roadmap for the development of Blaze.
It is split into multiple roadmaps for different functional
components. This document is presently a draft

Data Descriptors
================

The goal for data data descriptors is to solidify its design,
in particular how it uses dynd for in-memory array data.

In dynd, there is an array type/metadata/data abstraction which
is the basis of the array memory layout. This will serve a role
similar to the Python buffer protocol, with an ABI lockdown at
the point where the type system extensibility, completeness,
and stability are at the desired level.

In blaze, the data descriptor builds higher level abstractions,
for deferred evaluation, describing data from persistent formats
like BLZ and others, and describing data that is distributed
across multiple servers.

This is part of the foundation for execution, as executing
a blaze expression requires creating 

Release 0.3
-----------

* Modify the blaze data descriptor to use dynd as its memory
  representation instead of the current C-order/C-aligned
  subset.

Release 0.4
-----------

* Design how more advanced indexing and filtering interact with
  dynd and the data descriptor. How can blaze cleanly represent
  slicing, boolean indexing, and indirect indexing
  (from numpy, "fancy indexing").

Execution System
================

The goal for execution is to lay down a solid design and implementation
for elementwise kernel execution. This includes simple elementwise
operations with broadcasting, and reductions which can be computed element
by element (e.g. sum, but not sort). More complicated operations will
initially be treated like "generalized ufuncs" in numpy, as elementwise
operations can consume/produce array dimensions.

Each operation to be executed can be considered in three stages:

1. The interface provided by the Blaze library, on array/table objects.
2. The deferred representation of the operation. (Embodied by the
   KernelTree)
3. Execution of the operation on array data. This can further be split
   into execution on local memory data and distributed/out of core
   execution (which normally would build on the local memory execution).

All in-memory execution is done through the construction of a ckernel,
then calling the ckernel function pointer. The two main ways to get
a ckernel are using LLVM to JIT compile one, in blaze, or having
dynd assemble one from more primitive ckernels. Generally, the blaze
JIT compiling approach is best when the amount of data is large,
while the dynd ckernel composition approach is best when the amount
of data is small.

Out of core and distributed execution are done at a higher level,
by the blaze executive. For example, an out of core algorithm may
read chunks of a large BLZ input, writing computed results to a
new BLZ output. A distributed algorithm may orchestrate computations
on a number of blaze servers by scheduling tasks to them.

Release 0.3
-----------

* Change all KernelTree JIT compilation to target ckernels, remove
  other C function prototype targets. Note that this does not impose
  constraints at the LLVM interface level, it can still use simple
  by-value functions for primitive scalar functions.
* Teach KernelTree JIT to understand dynd dimension types, so it
  can JIT broadcasting across variable-sized dimensions.
* Design document for the deferred representation of blaze operations.
  This is shared between blaze and dynd. One major goal is to convert
  all implicit broadcasting of dimensions into explicit operations,
  so all mappings of dimensions are specified in the representation.

Release 0.4
-----------

Streaming Dimensions
====================

The goal for streaming dimensions is to design and implement
interfaces for processing streams of bytes and streams of
array elements. There is a start of a related interface in
dynd, for writing to arrays with byte/string types and
var_dim types respectively.

One of the reasons to do streaming dimensions is integration
with Python's binary/string stream protocol and iterator
protocols. An example flow of data enabled by streaming
dimensions is loading a binary .gz file with Python's gzip
module, then connecting it to blaze via a bytes array. This
could then be reinterpreted as an array of binary structures.
An eventual stretch goal would be to have this result flow
through some generator functions and expressions that blaze
could convert into native code using numba.

Release 0.3
-----------

* Define the bytes streaming dimension protocol for bytes/string.
* Define the var streaming dimension protocol for elements (like var_dim).
* Implement an adapter from a python file handle-like object
  (with read/readfrom/close methods) to bytes and string objects.
* Implement an adapter from a python iterator object to
  a blaze array.
* Consider how the bytes/string streaming dimension protocol
  should support an optional seek.
* Consider how the var streaming dimension protocol should
  optionally support restarting the sequence.  

Release 0.4
-----------

* Implement an adapter from blaze arrays to a python file handle-like
  object (with write/close methods).
* Implement an adapter from blaze arrays to the python iterator
  protocol, based on the var streaming dimension protocol.

Kernel Libraries
================

Blaze needs functions, both free functions and methods/properties
associated with blaze types like bool (any, all),
complex (real, imag, conjugate), string (find, strlen), etc.

Blaze has blaze functions, which is
the basis for this functionality. There needs to be a registration
system associating blaze functions with properties and
methods for particular dshapes.

There is a mechanism for exposing these in dynd objects,
including promoting methods and properties from scalars to array
dimensions. These dynd functions need to be liftable to
blaze functions.

Release 0.3
-----------

* Mechanism for associating methods and properties with a
  blaze scalar type, including promotion from scalar to
  array types.
* Lifting of dynd functions to blaze functions.
* Lifting of numpy ufuncs to dynd/blaze.

Release 0.4
-----------

Datashape System
================

One goal in the blaze datashape system is to work towards
representing simple ETL-style data transformations in a
simple way. We have experienced that many transformations
which are quite simple, cannot currently be quickly expressed in
a high performance way.

Another goal is to integrate datashapes more tightly with
the python type system. This is one of the difficulties identified
in numpy, where there is a dtype system independent of python
types along with a scalar type system.

Release 0.3
-----------

* Develop some ETL datashape prototypes, for example flexible string
  to date parsing.

Release 0.4
-----------

* Make dshape objects be python types, which create
  blaze arrays. (e.g. blaze.int32([1,2,3]) would create
  a 3-element array of 32-bit integers).

Missing Data
============

This goal is to introduce an NA missing value abstraction into both blaze
and dynd.

Error Handling
==============

An error handling mechanism consistent between dynd and blaze is needed
for the C ABI level of ckernels. Currently, dynd is using C++ exceptions,
but this isn't quite right for LLVM JIT generated code.

Another aspect of error handling is that it is common to desire for
"as much of possible" of a large array operation to succeed, with errors
accumulated separately. Having something like this be possible, without
adversely affecting performance when it's not needed, needs to be considered
to make sure blaze can evolve towards ideas in this vein.

