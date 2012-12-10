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

Two types of uses for data descriptors.

Case A: ( Numpy + Numba )
    How do I find the existing data

    What is the start pointer, what are the strides?

    In this case *all* the memory needed for the kernel to execute
    are available on the Blaze heap before execution begins.

Case B: ( Custom Python + C )
    How do I pull the data myself inside the kernel.

    Kernel needs a read() function which pulls bytes from the Source
    attached the datadescriptor

    In this case *none* of the memory is preloaded, the kernel allocates
    it as needed by loading it from disk, file, etc.


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
