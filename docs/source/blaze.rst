Requirements of Numba for Blaze
-------------------------------

* Irregular Arrays

    Ability to generate loops around chunked data arrays, where data is
    shuffled in and out of memory through various seeks and calls on
    data descriptor objects. Numpy & Minivect assume array is in memory
    at the time of execution, but the general case of Blaze is that the
    array is either on disk or from a remote source.

    Current Numba pipeline assumes that array objects essentially look like
    existing Numpy arrays but this is not an assumption that holds in Blaze.

    Numba doesn't need to support these cases itself ( that's what Blaze
    should do ) but at the same time the pipeline should be factorable
    enough that Blaze can use the Numba pipeline for codegen instead
    of rolling another codegen path.

* Types

    Blaze has it's own type inference engine what it calls "unification"
    for coercing datashape objects together. Numba kernel execution
    ideally shouldn't require use of Numpy types.

* Agnostic backend for code generation

    Blaze doesn't build AST or Python in the process of building an
    execution plan, it will generate a flexible graph and a collection of
    "blocks" of execution that dispatch in serial.

    Basically it amounts to very simple execution... much less
    complex than arbitrary Python code.
        
        store tmp1 call(kernel1 a b)
        store tmp2 call(kernel2 a b)
        store tmp3 call(kernel3 tmp2 tmp2)
        store tmp4 call(kernel4 tmp3 tmp3)
        store result tmp2
        ret result

    Blaze is capable of building up the storage and shuffling of bytes
    around but should invoke Numba kernels so as not to reimplement the
    same technolgoy as Numba.
