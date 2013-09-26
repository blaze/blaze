[TOC]

Blaze Deferred CKernel Interface
================================

 * [Blaze Execution System](blaze-execution.md)

This is a draft specification for a low level kernel
interface ABI. The purpose of this specification is
to provide a low level interface, just above the
[ckernel interface](ckernel-interface.md), which has
a little bit more flexibility to bind a given kernel
to dynd memory arrays with the same types but different
metadata. This also provides the mechanism to communicate
third party kernels to blaze and dynd.

Interface Goals
---------------

 * To provide a way to wrap third party functions, that
   can lower to different specific ckernel prototypes
   like strided or single, or with different dynd metadata,
   when the blaze execution system needs it.
 * Specific cases include wrapping numpy ufuncs/gufuncs,
   and being a target for blaze JIT compiling (to delay actual JIT
   until single or strided kernel is requested).

Deferred Kernel Definition
--------------------------

The deferred kernel is similar to the dynamic kernel instance
which wraps one particular ckernel. It is a by-value struct,
which owns an opaque allocation of memory. It has a little bit
of information about the kind of ckernel it generates, including
the size of the ckernel data,

```cpp
enum kernel_type_t {
    /** Kernel function unary_single_operation_t or expr_single_operation_t */
    kernel_request_single,
    /** Kernel function unary_strided_operation_t or expr_strided_operation_t*/
    kernel_request_strided
};

enum kernel_funcproto_t {
    unary_operation_funcproto_id,
    expr_operation_funcproto_id,
    binary_predicate_funcproto_id
};

/**
 * Function prototype for instantiating a ckernel from a
 * deferred_ckernel (dckernel). To use this function, the
 * caller should first allocate the appropriate
 * amount of memory (dckernel->ckernel_size) with the alignment
 * required (sizeof(void *)). When the data types of the kernel
 * require metadata, such as for 'strided' or 'var' dimension types,
 * the metadata must be provided as well.
 *
 * \param self_data_ptr  This is dckernel->data_ptr.
 * \param out_ckb  A ckernel_builder into which the ckernel is placed
 * \param ckb_offset  An offset within the out_ckb ckernel where to place it.
 * \param dynd_metadata  An array of dynd metadata pointers,
 *                       matching ckrenel->data_dynd_types.
 * \param kerntype  Either kernel_request_single or kernel_request_strided,
 *                  as required by the caller.
 *
 * \returns  The offset immediately after the created ckernel. This must
 *           be returned so ckernels with multiple children can place them
 *           one after another.
 */
typedef intptr_t (*instantiate_fn_t)(void *self_data_ptr,
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                const char *const* dynd_metadata, uint32_t kerntype);

struct deferred_ckernel {
    // A value indicating what type of ckernel this instantiates,
    // an enumeration value from kernel_funcproto_t.
    // 0: unary_single_operation_t/unary_strided_operation_t
    // 1: expr_single_operation_t/expr_strided_operation_t
    // 2: binary_single_predicate_t
    size_t ckernel_funcproto;
    // The number of types in the data_types array
    size_t data_types_size;
    // An array of dynd types for the kernel's data pointers.
    // Note that the builtin dynd types are stored as
    // just the type ID, so cases like bool, int float
    // can be done very simply.
    // This array should be part of the memory for data_ptr.
    const dynd::base_type * const* data_dynd_types;
    // A pointer to typically heap-allocated memory for
    // the deferred ckernel. This is the value to be passed
    // in when calling instantiate_func and free_func.
    void *data_ptr;
    // The function which instantiates a ckernel
    instantiate_fn_t instantiate_func;
    // The function which deallocates the memory behind data_ptr.
    void (*free_func)(void *self_data_ptr);
};
```

Simple Deferred CKernel Example
-------------------------------

To illustrate what this interface allows, let's start with a really
simple example of a deferred ckernel that defines an operation
which triples an int32. This example is defined statically,
it does not require memory management.

```cpp
// Kernel function which processes a single element
void int32_triple_single(char *dst, const char * const *src,
                kernel_data_prefix *extra)
{
    int32_t val = *(const int32_t *)src[0];
    val *= 3;
    *(int32_t *)dst = val;
}

// Kernel function which processes a strided array of elements
void int32_triple_strided(char *dst, intptr_t dst_stride,
                const char * const *src, const intptr_t *src_stride,
                size_t count, kernel_data_prefix *extra)
{
    const char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];

    for (size_t i = 0; i < count; ++i) {
        int32_t val = *(const int32_t *)src0;
        val *= 2;
        *(int32_t *)dst = val;

        dst += dst_stride;
        src0 += src0_stride;
    }
}

static intptr_t instantiate_triple_int32(void *self_data_ptr,
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                const char *const* dynd_metadata, uint32_t kerntype)
{
    // Make sure the ckernel builder has enough space
    intptr_t ckb_end = ckb_offset + sizeof(ckernel_prefix);
    ckernel_builder_ensure_capacity_leaf(out_ckb, ckb_end);
    ckernel_prefix *out_ckernel = (ckernel_prefix *)(out_ckb->data + ckb_offset);
    if (kerntype == kernel_request_single) {
        out_ckernel->function = (void *)&int32_triple_single;
    } else if (kerntype == kernel_request_strided) {
        out_ckernel->function = (void *)&int32_triple_strided;
    } else {
        // raise an error...
    }

    return ckb_end;
}

static void empty_free_func(void *self_data_ptr)
{
}

static const dynd::base_type *triple_data_types = {
    (const dynd::base_type *)dynd::int32_type_id,
    (const dynd::base_type *)dynd::int32_type_id
};

static deferred_ckernel triple_kernel_int32 = {
    // ckernel_funcproto
    expr_operation_funcproto_id,
    // ckernel_size
    sizeof(dynd::kernel_data_prefix),
    // data_types_size
    2,
    // data_dynd_types
    &triple_data_types,
    // data_ptr
    NULL,
    // instantiate_func
    &instantiate_triple_int32,
    // free_func
    &empty_free_func
};
```

Deferred CKernel Example With Additional Data
---------------------------------------------

To take the `triple` kernel one step further, lets make the
factor of three a parameter instead of hardcoded in the kernel.
To do this, the deferred ckernel will need an additional data item,
which gets placed in the ckernel during instantiation.

```cpp
struct constant_int32_multiply_kernel_extra {
    // A typedef for the kernel data type
    typedef constant_int32_multiply_kernel_extra extra_type;

    // All ckernels are prefixed like this
    dynd::kernel_data_prefix base;
    // The extra data goes after
    intptr_t factor;

    // The kernel function which executes the operation once
    static void single(char *dst, const char * const *src,
                    kernel_data_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        int32_t val = *(const int32_t *)src[0];
        val *= (int32_t)e->factor;
        *(int32_t *)dst = val;
    }

    // The kernel function which executes the operation
    // across strided arrays.
    static void strided(char *dst, intptr_t dst_stride,
                const char * const *src, const intptr_t *src_stride,
                size_t count, kernel_data_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        int32_t factor = (int32_t)e->factor;
        const char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];

        for (size_t i = 0; i < count; ++i) {
            int32_t val = *(const int32_t *)src0;
            val *= 2;
            *(int32_t *)dst = val;

            dst += dst_stride;
            src0 += src0_stride;
        }
    }
};

struct deferred_constant_int32_multiply_kernel_extra {
    intptr_t factor;
};

static intptr_t instantiate_deferred_constant_int32_multiply(void *self_data_ptr,
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                const char *const* dynd_metadata, uint32_t kerntype)
{
    deferred_constant_int32_multiply_kernel_extra *self =
            (deferred_constant_int32_multiply_kernel_extra *)self_data_ptr;
    intptr_t ckb_end = ckb_offset + sizeof(constant_int32_multiply_kernel_extra);
    constant_int32_multiply_kernel_extra *out_ckernel =
            (constant_int32_multiply_kernel_extra *)(out_ckb->data + ckb_offset);
    if (kerntype == kernel_request_single) {
        out_ckernel->base.function = (void *)&constant_int32_multiply_kernel_extra::single;
    } else if (kerntype == kernel_request_strided) {
        out_ckernel->base.function = (void *)&constant_int32_multiply_kernel_extra::strided;
    } else {
        // raise an error...
    }

    // Copy the multiplication factor to the ckernel
    out_ckernel->factor = self->factor;
    return ckb_end;
}

static const dynd::base_type *constant_int32_multiply_data_types = {
    (const dynd::base_type *)dynd::int32_type_id,
    (const dynd::base_type *)dynd::int32_type_id
};

/**
 * This function builds a deferred ckernel for multiplying
 * int32 values by the specified factor.
 */
void make_deferred_constant_int32_multiply_kernel(
                deferred_ckernel *out,
                intptr_t factor)
{
    // Indicates which ckernel function prototype this provides
    out->ckernel_funcproto = expr_operation_funcproto_id;
    // How big the ckernel to be generated is
    out->ckernel_size = sizeof(constant_int32_multiply_kernel_extra);
    // How many data types the kernel refers to
    out->data_types_size = 2;
    // The array of data types. In a more dynamic example,
    // this can point inside of the out->data_ptr memory.
    out->data_dynd_types = constant_int32_multiply_data_types;
    // Pointer to data owned by the deferred ckernel
    out->data_ptr = malloc(deferred_constant_int32_multiply_kernel_extra);
    (deferred_constant_int32_multiply_kernel_extra *)out->data_ptr->factor = factor;
    // The function to instantiate a ckernel
    out->instantiate_func = &instantiate_deferred_constant_int32_multiply;
    // The destructor function for the deferred ckernel.
    out->free_func = &free;
}
```

Using a Deferred CKernel
------------------------

It is a bit more involved to use a deferred ckernel than a ckernel,
because one generally has to check which function prototype it uses,
as well as which types the arguments have. We're going do a simplified
example which constructs a deferred ckernel, then instantiates it
in both the `single` and `strided` variants.

```cpp
void call_single(ckernel_deferred *ckd)
{
    // This example is for a unary expr_function
    assert(ckd->ckernel_funcproto == expr_operation_funcproto);
    assert(ckd->data_types_size == 2);

    // Create a ckernel_builder object
    // (In C++ libdynd, just "ckernel_builder ckb;", a normal object).
    ckernel_builder_struct ckb;
    ckernel_builder_construct(&ckb);

    // Instantiate the ckernel
    const char *meta[2] = {0, 0}; // Pointers to dynd metadata for the operands
    ckd->instantiate_func(ckd->data_ptr, &ckb, 0, meta, kernel_request_single);

    ckernel_prefix *ck = (ckernel_prefix *)ckb->m_data;

    // Get the kernel function
    expr_single_operation_t kfunc;
    kfunc = (expr_single_operation_t)ck->function;

    // Set up some sample data for the kernel
    int32_t dst_val = 0, src_val = 12;
    char *dst = (char *)&dst_val;
    const char *src = (const char *)&src_val;
    // Call the kernel
    kfunc(dst, &src, ck);
    printf("called kernel: %d -> %d\n", (int)src_val, (int)dst_val);

    // Destroy the ckernel instance
    ckernel_builder_destruct(&ckb);
}

void call_strided(deferred_ckernel *ck)
{
    // This example is for a unary expr_function
    assert(ckd->ckernel_funcproto == expr_operation_funcproto);
    assert(ckd->data_types_size == 2);

    // Create a ckernel_builder object
    // (In C++ libdynd, just "ckernel_builder ckb;", a normal object).
    ckernel_builder_struct ckb;
    ckernel_builder_construct(&ckb);

    // Instantiate the ckernel
    const char *meta[2] = {0, 0}; // Pointers to dynd metadata for the operands
    ckd->instantiate_func(ckd->data_ptr, &ckb, 0, meta, kernel_request_strided);

    ckernel_prefix *ck = (ckernel_prefix *)ckb->m_data;

    // Get the kernel function
    expr_strided_operation_t kfunc;
    kfunc = (expr_strided_operation_t)ck->function;

    // Set up some sample data for the kernel
    int32_t dst_val[3] = {0, 0, 0}, src_val[3] = {12, -5, 3};
    char *dst = (char *)&dst_val[0];
    const char *src = (const char *)&src_val[0];
    intptr_t src_stride = sizeof(int32_t);
    // Call the kernel
    kfunc(dst, sizeof(int32_t), &src, &src_stride, ck);
    printf("called kernel:");
    for (int i = 0; i < 3; ++i)
        printf("[%d]: %d -> %d\n", i, (int)src_val[i], (int)dst_val[i]);

    // Destroy the ckernel instance
    ckernel_builder_destruct(&ckb);
}

void example_deferred_ckernel_usage()
{
    // Construct the constant int32 multiplication kernel (previous example)
    deferred_ckernel dc;
    make_deferred_constant_int32_multiply_kernel(&dc, 13)

    // Use it to get a single kernel
    call_single(&dc);

    // Use it to get a strided kernel
    call_strided(&dc);

    // Destroy the deferred ckernel
    dc->free_func(dc->data_ptr);
    memset(dc, 0, sizeof(dc));
}
```