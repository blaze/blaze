================
Blaze Functions
================

Blaze Functions are the simplest calculation engine for 
Blaze Arrays.   These objects are similar in spirit to the 
ufunc objects of NumPy.   However, Blaze Functions have three
important differences: 

 1. Eleement Kernel arguments can be vectors (like generalized ufuncs)
 2. More types of data are supported (including structures)
 3. Lazy evaluation -- calling a ufunc builds up a graph.


Blaze Element Kernels
=====================

KernelTrees
===========
