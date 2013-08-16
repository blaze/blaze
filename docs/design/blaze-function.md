Blaze Functions
===============

 * [Blaze Execution System](blaze-execution.md)

Blaze functions are like numpy's `ufunc` and `gufunc` objects.
A blaze function typically contains a "dictionary" mapping from
type signatures to kernel functions. These kernel functions may
be deferred ckernels or LLVM functions, and the LLVM functions
may be for a specific type or generic and able to compile down
to ckernels for patterns of types.

TODO: Expand/develop the blaze function design.