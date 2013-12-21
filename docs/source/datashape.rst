Datashape
=========

.. highlight:: ocaml

Datashape is a generalization of ``dtype`` and ``shape`` into a micro
type system which lets us overlay high level structure on existing
data in Table and Array objects.

Overview
~~~~~~~~

Just like in traditional NumPy, the preferred method of implementing
generic vector operators is through ad-hoc polymorphism. Numpy's style
of ad-hoc polymorphism allows ufunc objects to have different behaviors
when "viewed" at different types. The runtime system then chooses an
appropriate implementation for each application of the function, based
on the types of the arguments. Blaze simply extends this specialization
to data structure and data layout as well as data type ( dtype ).

Many of the ideas behind datashape are generalizations and combinations
of notions found in Numpy:

.. cssclass:: table-bordered

+----------------+----------------+
| Numpy          | Blaze          |
+================+================+
| Broadcasting   | Unification    |
+----------------+----------------+
| Shape          |                |
+----------------+ Datashape      |
| Dtype          |                |
+----------------+----------------+
| Ufunc          | Gufunc         |
+----------------+----------------+

Units
-----

Datashape types that are single values are called **unit** types. They
represent a fixed type that has no internal structure. For example
``int32``.

In Blaze there are two classes of units: **measures** and
**dimensions**. Measures are units of data, while dimensions are
units of shape. The combination of measure and dimension in datashape
constructors uniquely describe the space of possible values or
**dataspace** of a table or array object.

Bit Types
~~~~~~~~~

The native bit types or **CType** objects are unit types standing for
unboxed machine types. These inherit the notation from NumPy.

.. cssclass:: table-striped

================ =========================================================
Bit type         Description
================ =========================================================
bool             Boolean (True or False) stored as a byte
int8             Byte (-128 to 127)
int16            Integer (-32768 to 32767)
int32            Integer (-2147483648 to 2147483647)
int64            Integer (-9223372036854775808 to 9223372036854775807)
uint8            Unsigned integer (0 to 255)
uint16           Unsigned integer (0 to 65535)
uint32           Unsigned integer (0 to 4294967295)
uint64           Unsigned integer (0 to 18446744073709551615)
float16          Half precision float: sign bit, 5 bits exponent,
                 10 bits mantissa
float32          Single precision float: sign bit, 8 bits exponent,
                 23 bits mantissa
float64          Double precision float: sign bit, 11 bits exponent,
                 52 bits mantissa
complex[float32] Complex number, represented by two 32-bit floats (real
                 and imaginary components)
complex[float64] Complex number, represented by two 64-bit floats (real
                 and imaginary components)
================ =========================================================


Blaze also adds a variety of bit-like types which are implemented
a subset of specialized storage and computation backends.

.. cssclass:: table-striped

==========  =========================================================
Bit type    Description
==========  =========================================================
string      Variable length UTF-8 string.
bytes       Variable length arrays of bytes.
json        Variable length UTF-8 string which contains JSON.
==========  =========================================================


Several python types are can be mapped directly on to datashape types:

.. cssclass:: table-striped

===========  =========================================================
Python type  Datashape
===========  =========================================================
int          int32
bool         bool
float        float64
complex      complex[float64]
str          string
unicode      string
bytes        bytes
bytearray    bytes
buffer       bytes
===========  =========================================================

String Types
~~~~~~~~~~~~

To Blaze, all strings are sequences of unicode code points, following
in the footsteps of Python 3. The default Blaze string atom, simply
called "string", is a variable-length string which can contain any
unicode values.

Endianness
~~~~~~~~~~

The data shape does not specify endianness, data types
are in native endianness when processed by Blaze functions.

Products
--------

A comma between two types signifies a product type. Product types
correspond to branching possibilities of types.

The product operator ``(,)`` is used to construct product types.
It is a type constructor of two arguments with a special infix
sugar.

Example::

    a, b

It is also left associative, namely::

    ((a, b), c) = a, b, c

The outer element a product type is referred to as a **measure**
while the other elements of the product are referred to as
**dimensions**.

.. image:: svg/type_expand.png
    :align: center

The product operator has the additional constraint that the first
operator cannot be a measure. This permits types of the form::

    1, int32
    1, 1, int32

But forbids types of the form::

    1, 1
    int32, 1
    int32, int32

There is a algebraic relation between product types and sum types
( discussed below ).

Fixed
~~~~~

The unit shape type is a dimension unit type. They are represented
as just integer values at the top level of the datatype. These are
identical to ``shape`` parameters in NumPy. For example::

    2, int32

The previous signature Is an equivalent to the shape and dtype of a
NumPy array of the form::

    ndarray(dtype('int32'), shape=(1,2))

A 2 by 3 matrix of integers has datashape::

    2, 3, int32

With the corresponding NumPy array::

    ndarray(dtype('int32'), shape=(2,3))

Constructors
~~~~~~~~~~~~

A type constructor is a parameterized type definition for specifying a
function which produces new types given inputs.

For example type constructor with no parameters has the base
kind ``(*)``, a type constructor with two parameters has kind ``(*
-> *)``.

By supplying a type constructor with one or more **concrete types**, new
**type instances** can be constructed and added to the system. Datashape
types that are comprised of multiple unit types are called **composite**
types. The product operator discussed above yields composite types.
Example::

    2, int32

Datashape types with free parameters in their constructor are called
**parameterized** types. Example::

    type SquareMatrix T = N, N, T

Datashape types without free parameters in their constructor are called
**alias** types, and are similar to ``typedef`` in C. Alias types don't
add any additional structure they just ascribe a new name. Example::

    type AliasType N = N, N, int32

Datashape types can be **anonymous** or labeled. Once a type is
registered it can be used in dshape expressions just like primitive
values and to construct even higher order types.

Blaze does not permit recursive type definitions.

Datashape types are split into three equivalence classes.

Records
~~~~~~~

Record types are ordered struct-like objects which hold a collection of
types keyed by labels. Records are also an in the class of **measure**
types. Records are sugard to look like Python dictionaries but
are themselves type constructors of variable number of type arguments.

Example 1::

    type Person = {
        name   : string;
        age    : int;
        height : int;
        weight : int
    }

Example 2::

    type RGBA = {
        r: int32;
        g: int32;
        b: int32;
        a: int8
    }

Records are themselves types declaration so they can be nested,
but cannot be self-referential:

Example 2::

    type Point = {
        x : int;
        y : int
    }

    type Space = {
        a: Point;
        b: Point
    }

Or equivalently::

    type Space = {
        a: { x: int; y: int };
        b: { x: int; y: int }
    }

Composite datashapes that terminate in record types are called
**table-like**, while any other terminating type is called
**array-like**.

Example of array-like::

    type ArrayLike = 2, 3, int32

Example of table-like::

    type TableLike = { x : int; y : float }


Type Variables
~~~~~~~~~~~~~~

**Type variables** a seperate class of types expressed as free variables
scoped within the type signature. Holding type variables as first order
terms in the signatures encodes the fact that a term can be used in many
concrete contexts with different concrete types.

Type variables that occur once in a type signature are referred to as
**free**, while type variables that appear multiple types are **rigid**.

For example the type capable of expressing all square two dimensional
matrices could be written as a combination of rigid type vars::

    A, A, int32

A type capable of rectangular variable length arrays of integers
can be written as two free type vars::

    A, B, int32

Sums
----

A **sum type** is a type representing a collection of heterogeneously
typed values. There are four instances of sum types in Blaze's type
system:

* :ref:`variant`
* :ref:`union`
* :ref:`option`
* :ref:`range`

.. _variant:

Variant
~~~~~~~

A **variant** type is a sum type with two tagged parameters ``left`` and
``right`` which represent two possible types. We use the keyword
``Either`` to represent the type operator. Examples::

    Either(float,char)
    Either(int32,float)
    Either({x: int}, {y: float})

..
    1 + B + C ...

.. _union:

Union
~~~~~

A **union** or **untagged union** is a variant type permitting a
variable number of variants. Unions behave like unions in C and permit a
variable number of heterogeneous typed values::

    Union(int8,string)

::

    Union(int8,int16,int32,int64)

..
    A + B + C ...

.. _option:

Option
~~~~~~

A Option is a tagged union representing with the left projection being
the presence of a value while the right projection being the absence of
a values. For example in C, all types can be nulled by using ``NULL``
reference.

For example a optional int field::

    Option(int32)

Indicates the presense or absense of a integer. For example a (``5,
Option int32``) array could be model the Python data structure:

::

    [1, 2, 3, na, na, 4]

Option types are only defined for type arguments of unit measures and
Records.

..
    1 + A

.. _range:

Range
~~~~~

Ranges are sum types over intervals of Fixed dimensions types.

Ranges are heterogeneously fixed dimensions within a lower and upper
bound.

Example 1::

    Range(1,5)

A single argument to range is assumes a lower bound of 0.

The set of values of integer arrays with dimension less than or equal to
1000 x 1000 is given by the datashape::

    Range(1000), Range(1000), int32

The lower bound must be greater than 0. The upper bound must be
greater than the lower, but may also be unbounded ( i.e. ``inf`` ).

..
    (1 + 1 + 1 .. + 1)

Stream
~~~~~~

Ranges are sum types over shape instead of data.

A case where a ``Range`` has no upper bound signifies a potentially infinite
**stream** of values. Specialized kernels are needed to deal with data
of this type.

..
    (1 + 1 + ...)


Numpy Compatability
~~~~~~~~~~~~~~~~~~~

FAQ
---

* How do I convert from Blaze DataShape to NumPy shape and
  dtype?:

.. doctest::

    >>> from blaze.datashape import dshape, to_numpy
    >>> ds = dshape("5, 5, int32")
    >>> to_numpy(ds)
    ((5, 5), dtype('int32'))

* How do I convert from Numpy Dtype to Datashape?:

.. doctest::

    >>> from blaze.datashape import dshape, from_numpy
    >>> from numpy import dtype
    >>> from_numpy((5,5), dtype('int32'))
    dshape("5, 5, int32")


* How do I convert from Blaze DataShape to CTypes?

