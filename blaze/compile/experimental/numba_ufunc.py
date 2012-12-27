def build_ufunc(context, graph):
    """
    Using Numba we can take ATerm expressions and build custom
    ufuncs on the fly if we have NumbaPro.

    ::
        a + b * c

    ::
        def ufunc1(op0, op1, op2):
            return (op0 + (op1 * op2))

    Which can be executed by the runtime through the
    ElementwiseLLVMExecutor. We stash it in the 'ufunc' parameter in
    the context. It's preferable to build these, otherwise it would
    involve multiple numpy ufuncs dispatches.

    ::
        %0 := ElemwiseLLVM[ufunc1](%a, %b, %c)

    """
    context = dict(context)

    # if no numbapro then just a passthrough
    if not context['have_numbapro']:
        return context, graph

    aterm_graph = context['aterm_graph']

    # Build the custom ufuncs using the ExecutionPipeline
    from blaze.engine import execution_pipeline

    # NOTE: the purpose of the execution pipeline is for every component to
    # cooperate, not just numba
    p = execution_pipeline.ExecutionPipeline()
    p.run_pipeline(context, aterm_graph)

    return context, graph
