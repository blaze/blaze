Datashape Binary Data Layout
============================

A binary layout specification is needed for
kernels and other code to be able to consume and
produce data of a particular datashape. This document
specifies a mapping of datashape to raw memory layout
which is the default used when no additional information
about the data is known.

Measures
--------

All types are in native byte order.

================   ====================================================
Datashape          Binary Data Layout
================   ====================================================
bool               One byte, 0 = False, 1 = True
int<N>             N/8 bytes, stored in two's complement.
uint<N>            N/8 bytes, unsigned integer.
float<N>           N/8 bytes, IEEE 754 binary<N> format. Note that
                   float128 means binary128, usually different from
                   the C compiler's long double type.
cfloat<N>          Two consecutive float<N> values.
string             Two consecutive pointers, pointing at the beginning
                   of and one past the end of a UTF-8 buffer.
bytes              Two consecutive pointers, pointing at the beginning
                   of and one past the end of a raw memory buffer.
json               Same as datashape "string", but the contents should
                   be validated JSON.
int                RESERVED for a future abitrary-sized "big integer"
                   type. Use the int<N> formats explicitly for fixed
                   integer sizes.
categorical(<X>)   An unsigned integer, one of uint8, uint16, or
                   uint32, with values limited to the range
                   [0, len(X)). The integer chosen is the smallest
                   which can store (len(X) + 1) values.
================   ====================================================

Dimensions
----------

================   ====================================================
Datashape          Binary Data Layout
================   ====================================================
<N>, <suffix>      (where N is an integer.) N instances of the binary
                   layout for the datashape <suffix>, one after
                   the other.
<N>, <suffix>      (where N is a symbolic TypeVar.) This datashape
                   does not have a binary layout
var, <suffix>      One pointer to a separate buffer, followed by
                   a pointer-sized integer (intptr_t) containing
                   the number of elements. The target of the pointer
                   should contain this many instances of the
                   datashape <suffix>, one after the other.
================   ====================================================

Records
-------

================   ====================================================
Datashape          Binary Data Layout
================   ====================================================
{f0: <ds0>; ...}   A C struct of all the binary layouts of datashapes
                   <ds0>, etc., packed exactly as the C compiler
                   would.
================   ====================================================

Missing Data
------------

The missing data types are exactly as measure for their
parameter, but one bitpattern has been sacrificed for NA.

=========================  ====================================================
Datashape                  Binary Data Layout
=========================  ====================================================
Option(bool)               NA is 0xff.
Option(int<N>)             NA is -2^(N-1).
Option(float<N>)           NA is 0x7ea2 (float16), 0x7f8007a2 (float32),
                           0x7ff00000000007a2 (float64),
                           0x7fff00000000000000000000000007a2 (float128).
Option(cfloat<N>)          The first (real) component of the complex number
                           is the float<N> NA.
Option(string)             NA is two NULL pointers.
Option(bytes)              NA is two NULL pointers.
Option(categorical(<X>))   NA is (2^N - 1), where the binary layout of
                           categorical(<X>) is uint<N>.
=========================  ====================================================


Memory Management Note
----------------------

Some of these data layouts, including "string", "bytes", "json",
and "var", specify pointers into separate data buffers. The
ownership of these buffers must be held by the data descriptor
exposing the data, in addition to ownership of the primary data.

Properties of Datashape Objects
-------------------------------

To support working with data of these data layouts, the objects
in blaze which represent datashapes have several properties.

All datashapes which have a binary layout as defined here
must have properties `c_itemsize` and `c_alignment`, which
are the number of bytes one element of the measure requires,
and what alignment it must satisfy. This alignment is the
same alignment C uses when packing the element into a struct.

Records have a property `c_offsets`, which is
a tuple of offsets to the starts of all the fields.

Array datashapes have a property `c_strides`,
which is just like the NumPy array strides, a byte offset
for incrementing by one along each dimension.

