Datashapes
==========

Datashape is a generalization of dtype and shape into a micro
type system which lets us describe the high-level structure of
NDTable and IndexArray objects.

There are primitive machine types which on top of you can build
composite and dimensionalized structures.

```python
uint32
int32
float
char
...
```

Fixed Dimensions
----------------

Fixed dimensions are just integer values at the top level of the
datatype. These are identical to ``shape`` parameters in numpy.

```python
2, int32
```

Is an length 2 array.

```python
array([1, 2])
```

A 2 by 3 matrix matrix has datashape:


```python
2, 3, int32
```

```python
array([[ 1,  2,  3],
       [ 4,  5,  6]])
```

Constructors
------------

A type constructor is higher type that produces new named types from
arguments given. These are called **alias** types, they don't add any
additional structure they just provide a new name.

```python
Dim2       = N, M
Quaternion = 4, 4, int32
```

Once the types are registered they can be used in dtype expressions just
like primitive values and also to construct even higher order types.

For example a record of quaternions.

```python
{x: Quaternion, y: Quaternion}
```

Notes. The free variables in a alias are unique only to the
constructor. For example composing the Dim2 constructor would
functionally create 4 variable dimensions *not* a 4 dimensional
structure with 2 rigid dimensions.

```python
Dim, Dim2, int32 = a, b, c, d, int32

# NOT THIS
Dim, Dim2, int32 = a, b, a, b, int32
```

Record Types
------------

Record types are struct-like objects which hold a collection
of types keyed by labels. For example a pixel:

```python
RGBA = {r: int32, g: int32, b: int32, a: int8}
```

Most importantly they are heterogenous.

Enumeration Types
-----------------

An enumeration provides a dimension which is specified by

```python
Lengths = {1,2,3}
```

```python
Lengths * 3
```

This would correspond to a variable triangular table of
the form:

```python
           3

         * * *
Lengths    * *
             *
```

Variable Length
---------------

The dual of the fixed length is a variable dimension which corresponds
to a unique yet undefined variable that is not specified.

```python
A, A, int32
```

```python
A, B, int32
```

The first example corresponds to a neccessarily square matrix of
integers while the later permits rectangular forms.

Ranged Length
-------------

Variable length types correspond where the dimension is not
known.

```python
ShortStrings = Var(0, 5)*char
```

For example ```5*ShortStrings``` might be a table of the form.

```python
foo
bar
fizz
bang
pop
```

Compounded variable lengths are **ragged tables**.

```python
Var(0,5), Var(0,5), int32
```

Would permit tables of the form:

```
1 2 3 7 1
  4 5 8 1
  3 1 9
    1
```

```
1 3 7
1 2
1
```

Stream Types
------------

A stream is a special case of ``Var`` where the upper bound is
infinity. It signifies a potentially infinite stream of elements.
```Stream(RGBA)``` might be stream of values from a photosensor. Where
each row represents a measurement at a given time.

```python
{ 101 , 202 , 11  , 32 }
{ 50  , 255 , 11  , 0 }
{ 96  , 100 , 110 , 0 }
{ 96  , 50  , 60  , 0 }
```

Union Types
-----------

A union is a set of possible types, of which the actual value
will be exactly one of.

```python
IntOrChar  = Union(int32, char)
StringLike = Union(char, string)
```

```python
Pixel = Union(
    {r: int32, g: int32, b: int32, a: int8},
    {h: int32, s: int32, v: int32},
)
```

Nullable Types
--------------

A value that or may not be null is encoded as a ``Either``
constructor.

```python
MaybeFloat = Either float nan
MaybeInt   = Either int32 nan
```

Function Types
--------------

** Work in Progress **

Function types are dimension specifiers that are encoded by
arbitrary logic. We only specify their argument types are
return types at the type level. The ``(->)`` is used to specify
the lambda expression.

For example a two dimensional table where an extra dimension is
added whose length is a range between the sizes of the first two.

```python
A, B, ( A, B -> Var(A, B) ), int32 
```

Pointer Types
-------------

** Work in Progress **

Pointers are dimension specifiers like machine types but where
the data is not in specified by value, but *by reference*. We use
adopt same notation as LLVM where the second argument is the
address space to reference.

Pointer to a integer in local memory:

```python
int32*
```

Pointer to a 4x4 matrix of integers in local memory

```python
*(4, 4, int32)
```

Pointer to a record in local memory

```python
*{x: int32, y:int32, label: string}
```

Pointer to integer in a shared memory segement keyed by 'foo'

```python
*int32 (shm 'foo')
```

Pointer to integer on a array server 'bar'

```python
*int32 (rmt array://bar)
```

Parametric Types
----------------

** Work in Progress **

The natural evolution is to support parametric types.

Which lets us have type constructors with free variables on the
left side of the constructor.

```python
# Constructor
Point T = {x: T, y: T}

# Concrete instance
Point int32 = {x: int32, y: int32}
```

Then can be treated atomically as a ``Point(int32)`` in programming
logic while the underlying machinery is able to substitute in the
right side object when dealing with concrete values.

For example, high dimensional matrix types:

```python
ctensor4 A B C D = A, B, C, D, complex64
```

```python
x = ctensor4 A B C D

rank x     = 4
outerdim x = A
innerdim x = D
```

And with a bit more machinery  allow parametric polymorphism.

```python
product :: (ctensor a b X) -> (ctensor X c d) -> (ctensor a b c d)
product a b = numpy.outer(a,b)
```
