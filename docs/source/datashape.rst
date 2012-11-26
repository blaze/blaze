Datashape
=========

.. highlight:: erlang

Datashape is a generalization of ``dtype`` and ``shape`` into a micro
type system which lets us overlay high level structure on existing
data in Table and Array objects that can inform better code
generation and scheduling.

Overview
~~~~~~~~

Just like in traditional NumPy the preferred method of implementing
generic vector operators is through ad-hoc polymorphism. Numpy's style
of ad-hoc polymorphism allows ufunc objects to have different behaviors
when "viewed" at different types. The runtime system then chooses an
appropriate implementation for each application of the function, based
on the types of the arguments. Blaze simply extends this specialization
to data structure and data layout as well as data type ( dtype ).

In fact many of the ideas behind datashape are generalizations and
combinations of notions found in Numpy:

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

Datashapes in Blaze do not form a hierarchy or permit subtyping. This
differs from type systems found in other languages like OCaml and Julia
which achieve a measure of type polymorphism through the construction of
hierarchies of types with an explicit pre-ordering.

Blaze favors the other approach in that types do not exist in a
hierarchy but instead are inferred through constraint generation at
"compile time". In addition it also permits a weakened version of
gradual typing through a dynamic type ( denoted ``?`` ) which will allow
a "escape hatch" in the type system for expressing types of values which
cannot be known until runtime.

The goal of next generation of vector operations over Blaze structures
aim to allow a richer and more declarative way of phrasing operations
over semi-structured data. While types are a necessary part of writing
efficient code, the ideal type system is one which disappears entirely!

Machine Types
~~~~~~~~~~~~~

+----------------+
| long           |
+----------------+
| double         |
+----------------+
| short          |
+----------------+
| longdouble     |
+----------------+
| char           |
+----------------+
| int8           |
+----------------+
| int16          |
+----------------+
| int32          |
+----------------+
| int64          |
+----------------+
| uint           |
+----------------+
| uint8          |
+----------------+
| uint16         |
+----------------+
| uint32         |
+----------------+
| uint64         |
+----------------+
| float16        |
+----------------+
| float32        |
+----------------+
| float64        |
+----------------+
| float128       |
+----------------+
| complex64      |
+----------------+
| complex128     |
+----------------+
| complex256     |
+----------------+
| void           |
+----------------+
| bool           |
+----------------+
| pyobj          |
+----------------+

Constructors
~~~~~~~~~~~~

Datashape types that are single values are called **unit** types. They
represent a fixed type that is not dependent on any parameters. For
example ``int32`` or ``2``.

Datashape types that are comprised of multiple unit types are
called **composite** types. Example::

    2, int32

Datashape types that are comprised of unbound free variables are called
**variadic** types. Example::

    A, B, int32

A type constructor is higher type that produces new named types from
arguments given. Example


Datashape types with free parameters in their constructor are called
**parameterized** types.

::

    SquareIntMatrix = N, N, int32

Datashape types wihtout free parameters in their constructor are called
**alias** types. They don't add any additional structure they just
provide a new name.

::

    SquareMatrix T = N, N, T

The ``int`` and ``float`` types are automatically aliased to the either
``int32`` or ``int64`` types depending on the platform.

Once the types are registered they can be used in dtype expressions just
like primitive values and also to construct even higher order types.
Blaze does not permit recursive type definitions at this time.

Datashape types are broken into three equivalence classes.

:Fixed:

    Fixed types are equal iff their value is equal.::

        1 == 1
        1 != 2

:CTypes:

    Machine types are equal iff their data type name and width
    are equal.::

        int32 == int32
        int64 != int32
        int8 != char

:Composite:

    Composite datashape types are **nominative**, in that the equivalence of
    two types is determined whether the names they are given are equivalent.
    Thus two datashapes that are defined identically are still not equal to
    each other.::

        A = 2, int32
        B = 2, int32

        A == A # True
        A == B # False

While it is true that structurally equivalent composites are not equal
to each other, it is however necessarily true that the unification of
two identically defined composite types is structurally identical to the
two types.

Fixed
~~~~~

Fixed dimensions are just integer values at the top level of the
datatype. These are identical to ``shape`` parameters in NumPy. ::

    2, int32

Is an equivalent to a Numpy array of the form::

    array([1, 2], dtype('int32'))

A 2 by 3 matrix of integers has datashape::

    2, 3, int32

With the corresponding NumPy array::

    array([[ 1,  2,  3],
           [ 4,  5,  6]])

Records
~~~~~~~

Record types are ordered struct-like objects which hold a collection of
types keyed by labels.

Example 1::

    Person = {
        name   : string,
        age    : int,
        height : int,
        weight : int
    }

Example 2::

    RGBA = {
        r: int32,
        g: int32,
        b: int32,
        a: int8
    }

Enumeration Types
-----------------

A enumeration specifies a number of fixed dimensions
sequentially::

    {1,2,4,2,1}, int32

The above could describe a structure of the form::

    [
        [1],
        [1,1],
        [1,1,1,1],
        [1,1],
        [1]
    ]

..
    (1 + 2 + 4 + 2 + 1) * int32

Variadic
~~~~~~~~

Variadic types expression unknown, but fixed dimensions which are scoped
within the type signature.

For example the type capable of expressing all square two dimensional
matrices could be written as::

    A, A, int32

A type capable of rectangular variable length arrays of integers
can be written as::

    A, B, int32


..
    (1 + 2 + ... + A) * (1 + 2 + ... B ) * int32

Ranges
~~~~~~

Ranges are unknown fixed dimensions within a lower and upper
bound.

Example 1::

    Var(1,5)

The lower bound must be greater than 0. The upper bound must be
greater than the lower, but may also be unbounded ( i.e. ``inf`` ).

A case where a range has no upper bound signifies a potentially infinite
**stream** of values. Specialized kernels are needed to deal with data
of this type.

..
    (a + ... + b) * int32


Tagged Union
~~~~~~~~~~~~

A tagged union is a sum type with two parameters ``left`` and
``right`` which represent the presence of two possible types::

    Either float char
    Either int32 na
    Either {1,2} {4,5}

..
    left, right
    forward, backward

Union
~~~~~

A union is syntactic sugar for repeated construction of application
composition of Either to a variable number of types. Unions behave like
unions in C and permit a collection of heterogeneous types within the
same context::

    Union int8 int16 int32 int64

This construction is always well-defined because of the associativity of
the sum type.

..
    A + B + C ...

Nullable
~~~~~~~~

Nullable types are composite types that represent the presence or
absence of a value of a specific type. Many languages have a natural
expression of this by allowing all or most types to be nullable
including including C, SQL, and Java.

For example a nullable int field::

    Either int32 null

..
    1 + A
