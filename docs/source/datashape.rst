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
gradual typing through a dynamic type ( denoted ``?`` ) which allows a
"escape hatch" in the type system for expressing types of values which
cannot be known until runtime.

The goal of next generation of vector operations over Blaze structures
aim to allow a richer and more declarative way of phrasing operations
over semi-structured data. While types are a necessary part of writing
efficient code, the ideal type system is one which disappears entirely!

Unit Types
~~~~~~~~~~

Datashape types that are single values are called **unit** types. They
represent a fixed type that has no internal structure. For example
``int32`` or ``2``.

In Blaze there are two classes of units **measures** and **dimensions**.
Measures are units of data, while dimensions are units of shape.

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
| bool           |
+----------------+
| pyobj          |
+----------------+

Products
--------

A comma between two types signifies a product type. Product types
correspond to branching possibilities of types.

The product operator ``(,)`` is used to construct product types.
It is a type constructor of two arguments.

Example::

    a, b

It is also left associative, namely::

    ((a, b), c) = a, b, c

The product operator has the additional constraint that the first
operator cannot be a measure. This permits types of the form::

    a, int32
    a, b, int32

But forbids types of the form::

    int32, a
    int32, int32

There is a algebraic relation between product types and sum types
( discussed below ).

Fixed
~~~~~

The unit shape type is a **dimension** unit type. They are represented
as just integer values at the top level of the datatype. These are
identical to ``shape`` parameters in NumPy. ::

    2, int32

Is an equivalent to the shape and dtype of a NumPy array of the form::

    array([1, 2], dtype('int32'))

A 2 by 3 matrix of integers has datashape::

    2, 3, int32

With the corresponding NumPy array::

    array([[ 1,  2,  3],
           [ 4,  5,  6]])

Constructors
~~~~~~~~~~~~

Datashape types that are comprised of multiple unit types are
called **composite** types. The product operator discussed above
yields composite types. Example::

A **type operator** is higher type that maps each choice of parameter to
a concrete type instance.

    2, int32

Datashape types that are comprised of unbound free variables are called
**variadic** types. Example::

    A, B, int32

Datashape types with free parameters in their constructor are called
**parameterized** types. Example::

    SquareMatrix T = N, N, T

Datashape types without free parameters in their constructor are called
**alias** types. Alias types don't add any additional structure they just
ascribe a new name. Example::

    SquareIntMatrix = N, N, int32

For example, the ``int`` and ``float`` types are automatically aliased
to the either ``int32`` or ``int64`` types depending on the platform.

Once the types are registered they can be used in dtype expressions just
like primitive values and also to construct even higher order types.

Blaze does not permit recursive type definitions.

Datashape types are split into three equivalence classes.

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

Records
~~~~~~~

Record types are ordered struct-like objects which hold a collection of
types keyed by labels. Records are also an in the class of **measure**
types.

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

Composite datashapes that terminate in record types are called
**table-like**, while any other terminating type is called
**array-like**.

Enumeration
-----------

A enumeration specifies a number of fixed dimensions sequentially. Example::

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

Variadic types expression unknown, but fixed dimensions which are
expressed as free variables scoped within the type signature. The
variable is referred to as **type variable** or ``TypeVar``.

For example the type capable of expressing all square two dimensional
matrices could be written as::

    A, A, int32

A type capable of rectangular variable length arrays of integers
can be written as::

    A, B, int32

..
    (1x + 2x + ... + Ax) * (1y + 2y + ... By)

Sums
----

A **sum type** is a type representing a collection of heterogeneously
typed values. There are four instances of sum types in Blaze's type
system:

* Variants
* Unions
* Options
* Ranges

Variants
~~~~~~~~

A **variant** type is a sum type with two tagged parameters ``left`` and
``right`` which represent two possible types. We use the keyword
``Either`` to represent the type operator. Examples::

    Either float char
    Either int32 na
    Either {1,2} {4,5}

..
    1 + B + C ...

Union
~~~~~

A **union** or **untagged union** is a variant type permitting a
variable number of variants. Unions behave like unions in C and permit a
collection of heterogeneous typed values::

    Union int8 int16 int32 int64

..
    A + B + C ...

Options
~~~~~~~

Option types are variant types with the null datashape as one of the
parameters, representing the presence of absence of a value of a
specific types. Many languages have a natural expression of this by
allowing all or most types to be nullable including including C, SQL,
and Java.

For example a nullable int field::

    Either int32 null

..
    1 + A

Ranges
~~~~~~

Ranges are sum types over shape instead of data.

Ranges are heterogeneously fixed dimensions within a lower and upper
bound.

Example 1::

    Var(1,5)

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

The difference between a stream and a TypeVar is that TypeVar are scoped
within the datashape expression whereas ``Stream`` objects are not.

..
    (1 + 1 + ...)
