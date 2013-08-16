[[_TOC_]]

Blaze Execution
===============

The blaze execution system takes array expressions, built up by
using blaze functions' deferred execution, and produces executable
code to evaluate them. It includes high level representations of the
expressions, designed to be transportable across a network, as well
as C ABI constructs for low level libraries to communicate
deferred and immediately executable kernels.

This document describes the design of the execution system, starting
from the lowest level and working its way up. At the lowest level
are primitive C ABI interfaces designed to be interoperable across
any library boundaries, including between systems using different
standard libraries. These low level interfaces are used by the
higher level systems as JIT compilation targets, and as a way to
import implementation kernels from outside of blaze.

The CKernel Interface
---------------------

 * [CKernel Interface Documentation](ckernel-interface.md)

The lowest level execution interface in blaze is the ckernel.
Any time an operation gets executed in blaze, it is first reduced
down into a ckernel, either via JIT compilation or assembling together
other ckernels, and then executed as a ckernel.

When passing ownership of a ckernel from one software component
to another, it gets wrapped as a dynamic kernel instance. This
is defined as a small C struct with some basic information about
the ckernel, as well as a method to free its resources.

At the ckernel level, all information about types and possible variations
about memory layout has been baked into the code and data that make
up the ckernel. All that is left is the ability to call the kernel function
and to free the resources associated with the ckernel. This means that
code using a ckernel can be quite simple, it just needs to know the ckernel's
function prototype, and have data pointers that it knows conforms to the
types baked into the ckernel, and it can execute it.

The Deferred CKernel Interface
------------------------------

 * [Deferred CKernel Interface Documentation](deferred-ckernel-interface.md)

One fundamental aspect of both blaze and dynd is deferred execution.
For supporting deferred and cached execution at the low level, just one
small step above the ckernel interface, is the deferred dynamic kernel.
This object provides a simple interface to building a ckernel whose
structure has already been determined up to the dynd type level, and
just needs dynd metadata and a kernel type (i.e. single or strided) to
build a ckernel.

One of the use cases driving the deferred dynamic kernel is to provide
individual kernels to blaze and dynd function dispatch. While many of
the functions provided by blaze will be JIT compiled LLVM bitcode, there
needs to also be a way to expose functions to blaze from external systems
which know little or nothing about blaze and dynd.

Blaze AIR JIT Compilation
-------------------------

 * [Blaze AIR Documentation](blaze-air.md)

The next level up from deferred ckernels are blaze expressions represented
in the blaze AIR (array intermediate representation). These expressions
get JIT compiled into deferred ckernels, a step which effectively binds
generic array code to particular dynd types.

Blaze Expression Lowering to Blaze AIR
--------------------------------------

The highest level of the blaze execution system is taking the interface
provided to users of blaze, which includes blaze functions, blaze array
constructors, and similar pieces, and lowers it to blaze AIR. This is
represented as deferred blaze arrays which contain blaze AIR and references
to the concrete blaze arrays of the input data.

Blaze Functions
---------------

 * [Blaze Function Use Cases](blazefunc-usecases.md)

Blaze functions are the user facing representation of functionality
in blaze. This is like the `ufunc`/`gufunc` of numpy. A blaze function
typically contains a "dictionary" mapping from type signatures to
kernel functions. These kernel functions may be deferred ckernels or
LLVM functions, and the LLVM functions may be for a specific type or
generic and able to compile down to ckernels for patterns of types.
