Blaze Function Use Cases
========================

 * [Blaze Execution System](blaze-execution.md)

This document goes through a number of blaze function use cases,
and is intended to be a representative sample. Included are notes
on how each use case affects the blaze datashape type system
and function dispatch.

Matrix Vector Multiplication
----------------------------

`matvec :: (M, N, T) (N, T) -> (M, T)`

T is a type for which the blaze functions `multiply(T, T)` and `add(T, T)` are implemented,
and satisfy the appropriate mathematical properties. (This leads to the question of
how precise we want to be about these things.)

Example types for `T`:
    real numbers, complex numbers: should dispatch to BLAS when reasonable
    integers
    integers modulo n
    matrices (e.g. 3x3 or 4x4 graphics transformation matrices)
    quaternions
    symbolic expressions (e.g. sympy)

There should be a general fallback implementation which works for
any T which implements multiplication and addition. For common types,
there should be BLAS or other library dispatch code which is optimized.

If someone registers a new type in Blaze, and has some code for efficiently
doing this operation, they should be able to add their kernel to the blaze
function and have it be called when the new type is used.

Sorting
-------

`sorted :: (N, T) -> (N, T)`

`T` is a type for which the blaze function `less(T, T)` is implemented, and
defines a strict weak ordering.

There should be a general sort implementation which uses the blaze function
`less`, along with the ability to provide optimized implementations for
specific common types.

We will need to choose whether `sorted` is a stable sort, like in Python,
or whether there are separate `sorted` and `stable_sorted` like in C++.

Partition Based On Indices
--------------------------

`partition_indexed :: (N, T) (M, integer) -> (M, var, T)`

`T` can be anything in this case, all it requires is to be copyable.

This function simply slices a one-dimensional array into a ragged
two dimensional array. In many cases, this may be implementable as
a view into the original data.

Eigenvalues
-----------

`eig :: (N, N, T) -> { eigval: (N, T); eigvec: (N, N, T) }`

Computes the eigenvalues and eigenvectors of the input matrix. This
is basically numpy.eig, but using a struct within the blaze type system
instead of a tuple from python's type system for the return value.

Noise Functions
---------------

`noise :: T -> T`
`         (2, T) -> T`
`         (3, T) -> T`

`curlnoise :: (2, T) -> (2, T)`
`             (3, T) -> (3, T)`

Here, `T` is a real type. It may make more sense to introduce a
vector type that doesn't act as an array dimension for broadcasting
purposes. In this case, the signatures may look more like the following.

`noise :: T -> T`
`         T[2] -> T[2]`
`         T[3] -> T[3]`

Gradient Function
-----------------

`gradient :: (N, T) -> (N, T)`
`            (M, N, T) -> (M, N, T[2])`
`            (M, N, R, T) -> (M, N, R, T[3])`

One version of this function would calculate the gradient of the function,
given a reconstruction filter. This function pokes at an ambiguity about
array dimensions and function dispatch vs broadcasting. Given an array
of datashape `(M, N, T)`, how should one choose between broadcasting the
`(N, T) -> (N, T)` signature, or using the signature `(M, N, T) -> (M, N, T[2])`?

Another version of this function might accept the gradient reconstruction function as
another blaze function, perhaps like this:

`gradient :: (N, T) ((T[3], real) -> T) -> (N, T)`
`            (M, N, T) ((T[3,3], real[2]) -> T[2]) -> (M, N, T[2]`
`            (M, N, R, T) ((T[3,3,3], real[3]) -> T[3]) -> (M, N, R, T[3]`

In this version of the function, the ambiguity is removed, because the dimensionality
of the reconstruction filter is provided in the type of the blaze function. On the other
hand, if the blaze function provided is itself a generic reconstruction filter which works
for any number of dimensions, this ambiguity still exists.

A big reason to prefer overloading this way instead of having multiple `gradient2`,
`gradient3`, etc. functions is that it is possible to write a lot of code using these
functions generically for both 2 and 3 dimensions.

Another possibility for separable reconstruction filters might be:

`gradient :: (N, T) (real -> real[3]) -> (N, T)`
`            (M, N, T) (real -> real[3]) (real -> real[3]) -> (M, N, T[2]`
`            (M, N, R, T) (real -> real[3]) (real -> real[3]) (real -> real[3]) -> (M, N, R, T[3]`

In this case, the number of filter coefficient functions disambiguates between
the different numbers of dimensions.

Hessian Function
----------------

`hess :: (M, N, T) -> (M, N, T[2,2])`

This function is like gradient, but now each element produced is a
matrix instead of a vector.

Date/Time Parsing
-----------------

`strptime :: string string -> datetime`

Struct Field Manipulation
-------------------------

These functions dive a bit more into some dynamic possibilities, where
the output type depends on the values of an input parameter instead of
its type

`fields :: {names...:types...} (N, string) -> {subsetnames...:subsettypes...}`

This function selects a subset of fields out of a struct.

This raises the question of how do we pattern match
variadically against input types like the struct `{names...:types...}`

`rename_fields :: {names...:types...} map<string, string> -> {newnames...:types...}`

This function renames a selection of fields, using the associative array
as the remapping. This could be an overload of `fields` as well, where the
type of the second parameter determines what happens.

There's a question here of how should we spell parameterized types
like the associative container `map<string, string>`.

Reductions With axis= Arguments
-------------------------------

Reductions which can be characterized elementwise are an important
class of functions for blaze. This includes min, max, average, standard
deviation, and similar functions. In numpy, these functions take two
optional parameters, `axis=` and `keepdims=`, which control how array
dimensions are handled.

One big difference for blaze over numpy is the addition of variable-sized
dimensions