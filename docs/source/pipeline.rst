========
Pipeline
========

Plan generation pipeline is a series of composable pass stages
which thread a context and graph object through to produce various
intermediate forms resulting in an execution plan.

The plan is a sequential series of instructions to concrete functions
calls ( ufuncs, numba ufuncs, Python functions ) for the runtime to
execute serially.

The pipeline is externally stateless, it does not modify the environment
that runs it.

::

                       Input
                         |
     +----------------------+
     |          pass 1      |
     +--------|----------|--+
            context     ast
              |          |
       postcondition     |
              |          |
       precondition      |
              |          |
     +--------|----------|--+
     |          pass 2      |
     +--------|----------|--+
            context     ast
              |          |
       postcondition     |
                         |
       precondition      |
              |          |
     +--------|----------|--+
     |          pass 3      |
     +--------|----------|--+
            context     ast
              |          |
       precondition      |
              |          |
              +----------+-----> Output


Passes
------

A pass always takes two arguments a ``context`` and a ``graph``. The
context object holds the internal state of the graph syntax tree between
passes.::

    def my_pass1(context, graph):

        # logic
        context['state'] = my_complex_state

        return context, graph


A pass can also be decorated with decorators to provide
**postcondition** and **precondition** checks for the input and output
of the pass::


    pre  = lambda context: True
    post = lambda context: True

    @ppass(pre=pre, post=post)
    def my_pass1(context, graph):
        pass


Custom Pipelines
----------------

The pipeline can then be formed out of a ``Pipeline`` initializer
which takes a sequential list of the passes that comprise it. ::

    my_pipeline = Pipeline([my_pass1,
                            my_pass2,
                            my_pass3,
                           ])

Invoking the pipeline can be done through the ``compile`` function which
is roughly equivalent to the following code::

    def compile(source, target=my_pipeline, **inputs):
        ctx, plan = target.run_pipeline(source)
        return plan
