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

CKernel Definition
------------------

A blaze ckernel is a chunk of memory which begins with
two function pointers, a ckernel function pointer and a
destructor. This is followed by an arbitrary amount of
additional memory owned by the ckernel.

The ckernel function prototype is context-specific,
it must be known by the caller from separate information.
It may be known because the ckernel is the result of
a request for a specific kind of assignment ckernel, or
for a binary predicate, for example.

The memory of a blaze ckernel must satisfy the following
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
struct ckernel_prefix;
typedef void (*destructor_fn_t)(ckernel_prefix *);

struct ckernel_prefix {
    void *function;
    destructor_fn_t destructor;
};

struct ckernel_data {
    ckernel_prefix base;
    /* Additional kernel data... */
};
```

An example ckernel function prototype is a single assignment ckernel,
which assigns one value from a source memory location
to a destination.

```cpp
typedef void (*unary_single_operation_t)(
                char *dst, const char *src,
                ckernel_prefix *extra);
```

Error Handling
--------------

Requires specification and implementation!

How errors are handled in ckernels needs to be defined. In DyND,
this is done with C++ exceptions, but this does not appear to be
reasonable in a cross-platform/cross-language way.

CKernel Builder
---------------

Blaze ckernels may be allocated on the stack or the heap,
and are usually contained within a ckernel_builder object.
This object is defined by libdynd, which exposes C constructor,
destructor, and operation functions for code using a ckernel.
The ckernel_builder object itself may be on the stack or on
the heap, as the code managing the object sees fit.

Note that this object is not copyable or relocatable, it must
be constructed and destructed from the same memory location.

```cpp
struct ckernel_builder {
    // Pointer to the kernel function pointers + data
    intptr_t *m_data;
    intptr_t m_capacity;
    // When the amount of data is small, this static data is used,
    // otherwise dynamic memory is allocated when it gets too big
    intptr_t m_static_data[16];
};
```

The pointer `m_data` points to a block of memory, guaranteed to
be aligned to `sizeof(void*)`, whose total size is `m_capacity`.
When initialized, `m_data` points to the data in `m_static_data`,
so small ckernels can be constructed with

CKernel Builder Functions From DyND
-----------------------------------

DyND's `dynd._lowlevel` namespace provides some functions to
construct and work with ckernel_builder objects. These functions
are in the form of `ctypes` function pointers.

When using the object, get the contained ckernel by directly
reading `m_data`, and its capacity by directly reading
`m_capacity`.

`void _lowlevel.ckernel_builder_construct(void *ckb)`

Given an appropriately sized (18 * sizeof(void *)) and
aligned (sizeof(void *)) buffer, this constructs a ckernel_builder
object in that memory.

`void _lowlevel.ckernel_builder_destruct(void *ckb)`

Given a memory pointer which was previously constructed with
`ckernel_builder_construct`, this destroys it, freeing any
memory it may own.

`void _lowlevel.ckernel_builder_reset(void *ckb)`

Given a `ckernel_builder` instance, this resets it to a state
equivalent to being newly constructed.

`int _lowlevel.ckernel_builder_ensure_capacity_leaf(void *ckb, intptr_t requested_capacity)`

Given a `ckernel_builder` instance, this ensures that the ckernel
builder has at least the requested capacity. Returns 0 on success and
-1 on failure. If it succeeds, it is guaranteed that the `m_capacity`
field of the ckernel builder is at least `requested_capacity`.

`int _lowlevel.ckernel_builder_ensure_capacity(void *ckb, intptr_t requested_capacity)`

This is just like the matching `*_leaf` function, but allocates extra
space of `sizeof(ckernel_prefix)`, intended for ckernel factories which
produce ckernels with children.

More Documentation
------------------

The proposed interface is also used in DyND.
The documentation for kernels there provide some
examples of how kernels can be constructed and how a kernel
factory API might look:

https://github.com/ContinuumIO/libdynd/blob/master/documents/ckernels.md
