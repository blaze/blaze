Blaze CKernel Interface
=======================

 * [Blaze Execution System](blaze-execution.md)

This is a draft specification for a low level kernel
interface ABI. The purpose of this interface is for
third-party code to build kernels for blaze, and for
blaze to build kernels usable by third-party code.
This is a work in progress, if you would like to
discuss this design, please email the public blaze
development mailing list at blaze-dev@continuum.io

Interface Goals
---------------

The goals of this interface are:

 * To communicate low level executable kernels between
   Blaze, DyND, Numba, and other third-party library code.
   Third-party code shouldn't have to have a deep
   understanding of the blaze system to provide
   or consume kernels.
 * To allow safe, correct usage across any ABI boundary,
   i.e. to not rely on linking to a particular C/C++
   standard library, the Python API, a common blaze API, etc.
 * To allow thread-safe execution. An execution engine
   should be able to call any kernel following this
   specification at the same time on many different threads.

This interface provides a common denominator binary ABI
format for kernels, similar to how '.o' or '.obj' files
provide a format for compiled code before it gets linked
together.

Kernel Definition
-----------------

A blaze kernel is a chunk of memory which begins with
two function pointers, a kernel function pointer and a
destructor. This is followed by an arbitrary amount of
additional memory owned by the kernel.

The kernel function prototype is context-specific,
it must be known by the caller from separate information.
It may be known because the kernel is the result of
a request for a specific kind of assignment kernel, or
for a binary predicate, for example.

The memory of a blaze kernel must satisfy the following
restrictions:

 * Its alignment must be that of a pointer, i.e.
   4 on 32-bit platforms and 8 on 64-bit platforms.

 * Its total size must be divisible by the alignment.

 * It must be relocatable using a memcpy.

   - It cannot contain pointers to other
     locations within itself (must use offsets for this).

   - It cannot assume an address within its memory
     is aligned greater than pointer alignment, because
     other code could memcpy it to a location which
     changes that alignment.

As a C struct, this looks like

```cpp
struct kernel_data_prefix;
typedef void (*destructor_fn_t)(kernel_data_prefix *);

struct kernel_data_prefix {
    void *function;
    destructor_fn_t destructor;
};

struct kernel_data {
    kernel_data_prefix base;
    /* Additional kernel data... */
};
```

An example kernel function prototype is a single assignment kernel,
which assigns one value from a source memory location
to a destination.

```cpp
typedef void (*unary_single_operation_t)(
                char *dst, const char *src,
                kernel_data_prefix *extra);
```

Error Handling
--------------

Requires specification and implementation!

How errors are handled in ckernels needs to be defined. In DyND,
this is done with C++ exceptions, but this does not appear to be
reasonable in a cross-platform/cross-language way.

Dynamic Kernel Instance
-----------------------

Blaze kernels may be allocated on the stack or the heap,
but for communicating kernels across library boundaries,
a standard way to pass the kernel and memory allocation
information is needed.

```cpp
struct dynamic_kernel_instance {
    kernel_data_prefix *kernel;
    size_t kernel_size;
    void (*free_func)(void *);
};
```

In this structure, 'kernel' is a pointer to the
kernel object as described above, 'kernel_size'
is the size of the data held in the kernel, and
'free_func' is the function to call for deallocating
the memory 'kernel' points to. By providing this function
explicitly, different libraries can use different
memory subsystems without any linking incompatibilities.

Example Usage
-------------

Here's a simple example of how third-party C code might call
the Blaze API to get a kernel, call the kernel, and free
the kernel data.

```cpp
/* Blaze API function that returns a unary single kernel */
int blaze_get_some_kernel_function(...,
                dynamic_kernel_instance *out_kernel);

int apply_blaze_kernel(char *dst, const char *src, ...)
{
    dynamic_kernel_instance k;
    unary_single_operation_t kfunc;

    /* Request a kernel from blaze */
    if (blaze_get_some_kernel_function(..., &k) < 0) {
        /* propagate error */
        return -1;
    }

    /* Get the kernel function pointer and call it */
    kfunc = (unary_single_operation_t)k.kernel->function;
    kfunc(dst, src, k.kernel)

    /* To free the kernel, we must destruct it AND free its memory */
    k.kernel->destructor(k.kernel);
    k.free_func(k.kernel);

    /* return success */
    return 0;
}
```

Example Kernel Factories
------------------------

The proposed interface is almost exactly that used in DyND
presently. The documentation for kernels there provide some
examples of how kernels can be constructed and how a kernel
factory API might look:

https://github.com/ContinuumIO/libdynd/blob/master/documents/kernels.md
