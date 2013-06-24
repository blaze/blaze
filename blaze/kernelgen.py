from contextlib import contextmanager
from collections import namedtuple
from llvm.core import Constant
import llvm.core as lc
import llvm_cbuilder.shortnames as C
from .py3help import reduce

@contextmanager
def position(builder, block):
    '''Temporarily move to the new block and return once the context closes.
    '''
    orig = builder.basic_block
    builder.position_at_end(block)
    yield
    builder.position_at_end(orig)

def compare_unsigned(builder, cmpstr, lhs, rhs):
    icmpmap = {
        '==': lc.ICMP_EQ,
        '!=': lc.ICMP_NE,
        '<' : lc.ICMP_ULT,
        '<=' : lc.ICMP_ULE,
        '>' : lc.ICMP_UGT,
        '>=' : lc.ICMP_UGE,
    }
    return builder.icmp(icmpmap[cmpstr], lhs, rhs)

def compare_signed(builder, cmpstr, lhs, rhs):
    icmpmap = {
        '==': lc.ICMP_EQ,
        '!=': lc.ICMP_NE,
        '<' : lc.ICMP_SLT,
        '<=' : lc.ICMP_SLE,
        '>' : lc.ICMP_SGT,
        '>=' : lc.ICMP_SGE,
    }
    return builder.icmp(icmpmap[cmpstr], lhs, rhs)


def debug(builder, msg, *args):
    mod = builder.basic_block.function.module
    val = lc.Constant.stringz(msg)
    gvar = mod.add_global_variable(val.type, 'debugstr.%x' % hash(msg))
    gvar.initializer = val
    gvar.global_constant = True

    charp = lc.Type.pointer(lc.Type.int(8))
    printfty = lc.Type.function(lc.Type.int(), [charp], var_arg=True)
    printf = mod.get_or_insert_function(printfty, name='printf')
    builder.call(printf,
                      [builder.bitcast(gvar, charp)] + list(args))


_loop_info = namedtuple('loop_info', ['entry', 'body', 'incr', 'end',
                                      'indices'])

@contextmanager
def loop_nest(builder, begins, ends, order=None, intp=C.intp, steps=None, dbg=False):
    '''Insert a N-dimension loop nest.

    Equivalent to:

        ax0 = order[-1]
        ax1 = order[-2]
        ax2 = order[-3]
        for i in range(begins[ax0], ends[ax0], steps[ax0]):
            for j in range(begins[ax1], ends[ax1], steps[ax1]):
                for k in range(begins[ax2], ends[ax2], steps[ax2]):
                    ...

    order:  order[-1] is the outermost axis. order[0] is the innermost axis.
    begins: list of llvm value for the start of the index for each axis.
    ends:   list fo llvm value for the end of the index for each axis.
    steps:  default to 1 for all axis.
    intp:   integer type for indexing.
    dbg:    boolean to enable debug mode; default to false.

    Returns a namedtuple of with entry = <entry block>, body  = <body block>,
        incr = <increment block>, end = <ending block>,
        indices = <list of index values>

    Note: When the context exits, the builder is at the end of the original
          basicblock.  It is user's responsibilty to add branch into
          the entry of the loop.

    '''

    # default steps to one
    if not steps:
        steps = [Constant.int(intp, 1) for _ in range(len(begins))]

    if not order:
        order = range(len(begins))

    # initialize blocks
    func = builder.basic_block.function
    orig = builder.basic_block
    entry = func.append_basic_block('loop.entry')
    body = func.append_basic_block('loop.body')
    incr = func.append_basic_block('loop.incr')
    end = func.append_basic_block('loop.end')

    ndim = len(order)

    if ndim == 0:
        with position(builder, entry):
            builder.branch(body)
        with position(builder, incr):
            builder.branch(end)
        with position(builder, body):
            yield _loop_info(entry=entry, body=body, incr=incr, end=end, indices=[])
        return

    cond = func.append_basic_block('loop.cond')

    outer_axis = order[-1] 

    #### populate loop entry ####
    with position(builder, entry):
        # sentry valid ranges
        valid = reduce(builder.and_, [compare_signed(builder, '<', s, e)
                                      for s, e in zip(begins, ends)])
        builder.cbranch(valid, cond, end)

    #### populate loop cond ####
    with position(builder, cond):
        # initialize indices
        indices = [builder.phi(intp) for _ in range(ndim)]
        for dim, (ind, ibegin) in enumerate(zip(indices, begins)):
            ind.name = 'index.%d' % dim
            ind.add_incoming(ibegin, entry)
        # check if indices has ended
        pred = compare_signed(builder, '<', indices[outer_axis], ends[outer_axis])
        builder.cbranch(pred, body, end)

    #### populate loop body ####
    with position(builder, body):
        if dbg:
            fmt = '[%s]\n' % ', '.join(['%lld'] * ndim)
            debug(builder, fmt, *indices)
        yield _loop_info(entry=entry, body=body, incr=incr,
                         end=end, indices=indices)

    #### populate loop increment ####
    lastaxes = []
    nextbb = incr

    remain = [(ax, indices[ax]) for ax in reversed(order)]
    while remain:
        ax, ind = remain.pop()
        with position(builder, nextbb):
            for lastax, lastval in lastaxes:
                indices[lastax].add_incoming(lastval, builder.basic_block)

            indnext = builder.add(ind, steps[ax])
            pred = compare_signed(builder, '<', indnext, ends[ax])
            ind.add_incoming(indnext, builder.basic_block)

            nextbb = func.append_basic_block('incr_%d' % ax)
            builder.cbranch(pred, cond, nextbb)

            lastaxes.append((ax, begins[ax]))

            for ax, ind in remain:
                ind.add_incoming(ind, builder.basic_block)
    else:
        with position(builder, nextbb):
            builder.branch(end)

    #### position back to the original block ####
    assert builder.basic_block is orig

