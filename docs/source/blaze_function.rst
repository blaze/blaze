================
Blaze Functions
================

Blaze Functions are the simplest calculation engine for 
Blaze Arrays.   Simple operations like add, multiply, sin, 
cos, comparisons, etc. are implemented with Blaze Functions.
But, also complicated operations like eigenvalue decomposition,
dot-products, and Fourier Transforms can be implemented via Blaze
Functions.  Blaze Functions are any operation that is performed 
on an "element" where all the inputs and the output fit in 
memory.   Blaze Functions are like ufunc objects in NumPy 
with three important differences:

 1. Kernel arguments can be vectors (like generalized ufuncs)
 2. More types of data are supported (including structures)
 3. Lazy evaluation -- calling a ufunc builds up a graph and
    machine code is generated at evaluation time. 

Example: 

Assume a and b are Blaze Arrays or objects that can be converted
to Blaze Arrays and dot, mul, and add are Blaze Functions

result = dot(add(a,b), mul(a,b))

Result will be a "deferred" Blaze Array.  The eval function
must be called to create a materialized, concrete, or 
reified Blaze Array which will apply a generated kernel 
based on the expression.   The kernel is composed of the kernels
selected from add, mul, and dot based on the data-shape of
the input arguments.

