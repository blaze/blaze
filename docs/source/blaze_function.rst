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

Result will be a "deferred" Blaze Array.  The eval method 
must be called to create a materialized, concrete, or 
reified Blaze Array which will apply a generated kernel 
based on the expression.   The kernel is composed of the kernels
selected from add, mul, and dot based on the data-shape of
the input arguments.   These low-level kernels are built up 
using LLVM and compiled to machine code 
(using vector-instructions where possible).  
This fused, low-level kernel is applied over the data. 


Blaze Element Kernels
=====================
A Blaze Element Kernel is an object that contains an LLVM
function with a particular signature.  This function should 
perform the operation on a single element.  So, for example 
an add Element Kernel would wrap around an LLVM function that 
adds two typed values.   You don't need to worry about this 
being slow, the BlazeFunction compiler will create
an in-line version of the function.  The LLVM function
signature consists of the inputs followed by one output. 
(Multiple outputs can be returned from an element kernel by
using a structure as the output data-type.) 

Here is an example of an LLVM Function that can be used to build
a BlazeElementKernel: 

def add(a,b):
    return a + b

Using Numba or another Python to LLVM tool, this can be converted into the following LLVM function for the float64 data:

define double @add(double %a, double %b) nounwind readnone alwaysinline {
entry:
  %0 = fadd double %a, %b
  ret double %0
}


KernelTrees
===========
Evaluating BlazeFunctions create KernelTrees which are tree structures of BlazeElementKernels.  These tree structures are used to manage the stitching together and "lifting" of the Element kernels at run-time

Lifting
=======
The concept of lifting is that a BlazeElementKernel is "lifted" to become a kernel accepting array arguments with a larger number of elmeents. 


Broadcasting
============ 
