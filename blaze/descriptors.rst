Data Descriptors
================

DataDescriptors are the underlying, low-level references to data
that are used to inform how a kernel executes and loads data
during execution.

- Python implementation
- C Implemention
- Need mappings to go between the two

Functions:

   asbuffer     : function
   asbuflist    : function
   asstream     : function
   asstreamlist : function

Signatures::

    +-------------+
    |   Runtime   |
    +-------------+
     A |      ^
       v      | B
    +-------------+
    |   Kernel    |
    +-------------+

The Kernel knows about the algorithm, and hence knows what data access pattern
will give the best locality. Dimensional information needs to be available to
the kernel.

Then:

    The runtime handling reading and writing to the source. The runtime (the data
    descriptor) further exposes data descriptors that can return bulk data in the
    form of multi-dimensional tiles or chunks. Special cases such as 1D chunks simply
    have extent 1 in other dimensions. A special case like a NumPy array simply
    returns the entire array as the multi-dimensional chunk.

    Kernels pull data by specifying item indices, for which the data descriptor
    computes the right chunk (if chunking is used), and computes the remaining
    shape and the multi-dimensional strides to be used for computation on that chunk.

    After writing to a chunk, the chunk must be committed to allow write-back, compression,
    etc.

| Two types of uses for data descriptors.
|
| Case A: ( Numpy + Numba )
|     How do I find the existing data
|
|     What is the start pointer, what are the strides?
|
|     In this case *all* the memory needed for the kernel to execute
|     are available on the Blaze heap before execution begins.
|
| NOTE: Case A) is a special case of case B). A chunk iterator with a single iteration would suffice.
|
| Case B: ( Custom Python + C )
|     How do I pull the data myself inside the kernel.
|
|     Kernel needs a read() function which pulls bytes from the Source
|     attached the datadescriptor
|
|     In this case *none* of the memory is preloaded, the kernel allocates
|     it as needed by loading it from disk, file, etc.
|
| NOTE: read() is suitable for some operations, but not things like reductions or anything requiring
|       some sense of dimensionality

Matrix multiplication::

    i = 0
    while (i < ddescA->shape[0]) {
        j = 0
        while (j < ddescB->shape[1]) {
            k = 0;

            ddescC->read_chunk(&chunkC, i, j)
            while (k < ddescA.shape[1]) {
                ddescA->read_chunk(&chunkA, i, k)
                ddescB->read_chunk(&chunkB, k, j)


                // NOTE: chunk->shape[i] does not extend beyond shape[i]
                end_i = min(chunkA->shape[0], chunkC.shape[0])
                end_j = min(chunkB->shape[1], chunkC->shape[1])
                end_k = min(chunkA->shape[1], chunkB->shape[0])

                matmul(chunkA->data, chunkB->data, chunkC->data, end_i, end_j, end_k)
                chunkC->commit()

                i = end_i
                j = end_j
                k = end_k
            }
        }
    }

    void matmul(float \*A, float \*B, float \*C, end_i, end_j, end_k) {
        /* Tiled matmul \*/
        for (i0 = 0; i0 < B; i0 += B)
            for (j0 = 0; j0 < B; j0 += B)
                for (k0 = 0; k0 < B; k0 += B)
                    for (i = i0; i < min(i0 + B, end_i); i++)
                        for (j = i0; j < min(j0 + B, end_j); j++)
                            for (k = k0; k < min(k0 + B, end_k); k++)
                                // Use pointer arithmetic and strength reduction
                                C[i, j] += A[i, k] * B[k, j];
    }


NOTE: I think data should be pulled and computed on demand in all cases.

Case A
======

Two core arguments for *each* operand in kernel execution:

    ddesc  - DataDescriptor struct
    dshape - Datashape struct

**datashape**

Inside of the dshape will be tradition NumPy ufunc args as members. These will
apply for datashapes that are array-like.

For the contigious NumPy case:

::
    int *dimensions
    int *steps
    int **shape
    int **shape

For chunked array objects

::
    int *dimensions
    int *steps
    int **shape
    int **chunksize
    int **nchunks

For more exotic cases ( full datashape grammer ), we encode
datashape dimensions in C and let the algorithm access them to
specialize as needed.

::
    type_t* ty
    int array_like
    int table_like

**ddesc**

Inside of the ddesc will be the locations in memory for the
inputs and outputs and possibly some information about bounds.

::
    void **args
    void *out


Psueocode ::

    void unary_op(ddesc *dd, dshape *ds) {
        int *dimensions = ds->dimensions;

        char *input_1 = (char*)dd->args[0];
        char *input_2 = (char*)dd->args[1];
        char *output = (char*)out->args[2];

        int i;

        for (i = 0; i < dimensions[0]; ++i) {
            *output = CUSTOM_KERNEL(*input_1, *input_2);

            input_1 += ds->steps[0];
            input_2 += ds->steps[1];
            output  += ds->steps[2];
        }

    }

Case B
======

TODO, it passes function pointers in... write tomorrow
