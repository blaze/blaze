# -*- coding: utf-8 -*-

"""
Defines the Pipeline class which provides a series of transformation
passes on the graph which result in code generation.
"""
from functools import wraps

from blaze.plan import BlazeVisitor, InstructionGen
from blaze.compile.toposort import topovals
#from blaze.type_reconstruction import infer

debug = True

#------------------------------------------------------------------------
# Pipeline Combinators
#------------------------------------------------------------------------

def compose(f, g):
    return lambda *x: g(*f(*x))

def braid(f):
    return lambda *y: lambda *x: f(*x)(*y)

def bind(self, f, x):
    if x is None:
        return None
    else:
        if f(x):
            return x
        else:
            return None

#------------------------------------------------------------------------
# Pre/Post Conditions
#------------------------------------------------------------------------

# vacuously true condition
Id = lambda p: lambda x: x

#------------------------------------------------------------------------
# Passes
#------------------------------------------------------------------------
#
#                  Input
#                     |
# +----------------------+
# |          pass 1      |
# +--------|----------|--+
#        context     ast
#          |          |
#   postcondition     |
#          |          |
#   precondition      |
#          |          |
# +--------|----------|--+
# |          pass 2      |
# +--------|----------|--+
#        context     ast
#          |          |
#   postcondition     |
#                     |
#   precondition      |
#          |          |
# +--------|----------|--+
# |          pass 3      |
# +--------|----------|--+
#        context     ast
#          |          |
#   precondition      |
#          |          |
#          +----------+-----> Output

def ppass(pre=Id,post=Id):
    @wraps
    def outer(fn):
        return fn
    return outer

def do_environment(context, graph):
    context = dict(context)

    # TODO:

    return context, graph

def do_convert_to_aterm(context, graph):
    """Convert the graph to an ATerm graph
    See blaze.aterm

    ::
        a + b

    """
    context = dict(context)
    #vars = topovals(graph)

    # walk the blaze Graph objects ( Python objects inherting
    # derived expr.node.Node ) map them into a ATerm expression
    visitor = BlazeVisitor()
    aterm_graph = visitor.visit(graph)
    operands = visitor.operands

    # ----------------------
    context['operands'] = operands
    context['aterm_graph'] = aterm_graph
    # ----------------------

    return context, graph

def do_types(context, graph):
    context = dict(context)

    # Build the constraint graph and the environement mapping

    #cgraph, env = infer(graph)

    # Expand all type variable references in expressions with the actual
    # type instances in the context, local to the subexpression

    #graph = expand(cgraph, env)

    # ----------------------
    #context['type_env'] = env
    # ----------------------

    return context, graph

def debug_aterm(context, graph):
    if debug:
        print context['aterm_graph']

    return context, graph

def build_operand_dict(context, aterm_graph):
    """
    Map input operands ids to the input operands in context['operand_dict']
    """
    operands = context['operands']
    operand_dict = dict((id(op), op) for op in operands)
    context['operand_dict'] = operand_dict
    return context, aterm_graph

def substitute_llvm(context, aterm_graph):
    "Substitute executors for the parts of the graph we can handle"
    from blaze.compile.llvm import llvm_execution

    executors = {}
    context['executors'] = executors
    operand_dict = context['operand_dict']

    aterm_graph = llvm_execution.substitute_llvm_executors(
                        aterm_graph, executors, operand_dict)
    return context, aterm_graph

def do_plan(context, graph):
    """ Take the ATerm expression graph and do inner-most evaluation to
    generate a linear sequence of instructions from that together with
    the table of inputs and outputs, built kernels forms the execution
    plan.

    Example::

    ::
        a + b * c

    ::
        vars %a %b %c
        %0 = Elemwise[np.mul,nogil](%b, %c)
        %0 = Elemwise[np.add,nogil,inplace](%a, %0)
        ret %0

    """
    context = dict(context)

    aterm_graph = context['aterm_graph']
    executors = context['executors']

    igen = InstructionGen(executors)
    igen.visit(aterm_graph) # effectful
    plan = igen.plan # instructions
    vars = igen.vars
    symbols = igen.symbols

    context['plan'] = plan
    context['symbols'] = symbols

    return context, graph

#------------------------------------------------------------------------
# Pipeline
#------------------------------------------------------------------------

class Pipeline(object):
    """
    Plan generation pipeline is a series of composable pass stages
    which thread a context and graph object through to produce various
    intermediate forms resulting in an execution plan.

    The plan is a sequential series of instructions to concrete
    functions calls ( ufuncs, numba ufuncs, Python functions ) for the
    runtime to execute serially.
    """

    def __init__(self, passes, inputs=None):
        self.init = inputs or {}

        # sequential pipeline of passes
        self.pipeline = reduce(compose, passes)

    def run_pipeline(self, graph, plan=False):
        """
        Run the graph through the pipeline
        """
        # Fuse the passes into one functional pipeline that is the
        # sequential composition with the intermediate ``context`` and
        # ``graph`` objects threaded through.

        # pipeline = stn ∘  ... ∘  st2 ∘ st1

        context, plan = self.pipeline(self.init, graph)
        return context, context['plan']
        #return context, plan

    def __call__(self, graph):
        return self.run_pipeline(graph)

blaze_rts = Pipeline([do_environment,
                      do_convert_to_aterm,
                      do_types,
                      debug_aterm,
                      build_operand_dict,

                      # Codegen stages
                      substitute_llvm,
                      do_plan,
                      ])

def compile(source, target=blaze_rts, **inputs):
    ctx, plan = target.run_pipeline(source)
    return plan

def _compile(source, target=blaze_rts, **inputs):
    ctx, plan = target.run_pipeline(source)
    return ctx, plan
