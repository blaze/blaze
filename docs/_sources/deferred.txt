==================
Deferred Execution
==================

Every operator in blaze is expressed as a typed function. Every operation is
the application of an operator. And operand is then simply an array input, or
another deferred operation.

Under deferred evaluation, every operation that is executed is recorded in
a graph. Below we will describe the process end-to-end.

Blaze Functions
~~~~~~~~~~~~~~~

A blaze function (`blaze.function.BlazeFunc`) representing a high-level
operator. The function carries around different typed (python) functions.
The right function is selected through an overloading process, which selects
the "best match" from the input types.

With each typed function (which we shall call `overload`), we can
associate implementations (or `kernels):

    - A single (concrete) overload may have implementations associated of
      different `kinds`
    - With each kind we can associate several implementation "objects". The
      semantics for these objects depend on the `kind`

An implementation `kind` is simply a string, such as `python`, `numba`, `llvm`,
`ctypes`, and so forth.

To clarify pictorially:

.. digraph:: blazefunction

    "blaze function" -> "overload 1"
    "blaze function" -> "overload 2"
    "overload 1" -> "signature 1"
    "overload 2" -> "signature 2"
    "overload 1" -> "[('llvm', [llvm_func1, ...]), ('numba', [...])]"


Blaze functions are most often constructed through one of the decorators
in `blaze.function`:

    - ``@elementwise(signature)``: create an element-wise function
    - ``@function(signature)``: create a blaze-function

The element-wise decorator creates a function that acts strictly over
individual elements. The function decorator creates a function that
accepts whatever it signature says it should.

The `implement` method of a blaze function can be used to associate
implementations with the different overloads.

Expression Graph
~~~~~~~~~~~~~~~~

The expression graph is constructed whenever a blaze function is applied.
This performs incremental type inference, overload selection and type checking.
The graph is greated by `blaze.expr.construct`. Nodes consist of kernel
application or nodes representing inputs:

    - ``KernelOp(dshape, args, metadata)``
    - ``ArrayOp(dshape)``

A ``KernelOp`` further holds on to metadata that indicates which function
was called, and which overload was selected.

Note how the expression graph has no reference to the original arrays! Our
implementation lifts the compile-time values to runtime-time values (it
"forgets" them). This allows a clean separation between the compilation
engine and the evaluation engine. It also allows us, or external engines to
compile symbolic expressions.

Compilation & AIR
~~~~~~~~~~~~~~~~~

The next stage is compilation, which takes a deferred expression graph and
constructs blaze AIR (Array Intermediate Representation). AIR follows are
very similar representation to the expression graph, but is instead encoded
in a linear representation of 3-ary address code: every operation is an
instruction with a (result, opcode, args).

AIR is then interpreted by a sequence of interpreters according to the
evaluation strategy. There may be several passes, any of which can transform
the IR.

TODO: Work out more of this and document
