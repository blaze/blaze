"""
The purpose of this module is to create blaze functions. A Blaze Function
carries a polymorphic signature which allows it to verify well-typedness over
the input arguments, and to infer the result of the operation.

Blaze function also create a deferred expression graph when executed over
operands. A blaze function carries default ckernel implementations as well
as plugin implementations.
"""

from __future__ import print_function, division, absolute_import

from collections import namedtuple

# TODO: Remove circular dependency between blaze.objects.Array and blaze.compute
import blaze
import datashape
from datashape import coretypes, dshape
from datashape import coretypes as T

from ..datadescriptor import DeferredDescriptor
from .expr import ArrayOp, ExprContext, KernelOp

Overload = namedtuple('Overload', 'resolved_sig, sig, func')

def construct(bfunc, ctx, overload, args):
    """
    Blaze expression graph construction for deferred evaluation.

    Parameters
    ----------
    bfunc : Blaze Function
        (Overloaded) blaze function representing the operation

    ctx: ExprContext
        Context of the expression

    overload: blaze.overload.Overload
        Instance representing the overloaded function

    args: list
        bfunc parameters
    """
    assert isinstance(bfunc, BlazeFunc), bfunc

    params = [] # [(graph_term, ExprContext)]

    # -------------------------------------------------
    # Build type unification parameters

    for i, arg in enumerate(args):
        if isinstance(arg, blaze.Array) and arg.expr:
            # Compose new expression using previously constructed expression
            term, context = arg.expr
            if not arg.deferred:
                ctx.add_input(term, arg)
        elif isinstance(arg, blaze.Array):
            term = ArrayOp(arg.dshape)
            ctx.add_input(term, arg)
            empty = ExprContext()
            arg.expr = (term, empty)
        elif not isinstance(arg, blaze.Array):
            term = ArrayOp(T.typeof(arg))

        ctx.terms[term] = arg
        params.append(term)

    assert isinstance(overload.resolved_sig, T.Function)
    restype = dshape(overload.resolved_sig.restype)

    return KernelOp(restype, *params, kernel=bfunc, overload=overload)


class BlazeFunc(object):
    """
    Blaze function. This is like the numpy ufunc object, in that it
    holds all the overloaded implementations of a function, and provides
    dispatch when called as a function. Objects of this type can be
    created directly, or using one of the decorators like @function .

    Attributes
    ----------
    overloader : datashape.OverloadResolver
        This is the multiple dispatch overload resolver which is used
        to determine the overload upon calling the function.
    ckernels : list of ckernels
        This is the list of ckernels corresponding to the signatures
        in overloader.
    plugins : dict of {pluginname : (overloader, datalist)}
        For each plugin that has registered with this blaze function,
        there is an overloader and corresponding data object describing
        execution using that plugin.
    name : string
        The name of the function (e.g. "sin").
    module : string
        The name of the module the function is in (e.g. "blaze")
    fullname : string
        The fully qualified name of the function (e.g. "blaze.sin").

    """

    def __init__(self, module, name):
        self._module = module
        self._name = name
        # The ckernels list corresponds to the
        # signature indices in the overloader
        self.overloader = datashape.OverloadResolver(self.fullname)
        self.ckernels = []
        # Each plugin has its own overloader and data (a two-tuple)
        self.plugins = {}

    @property
    def name(self):
        """Return the name of the blazefunc."""
        return self._name

    @property
    def module(self):
        return self._module

    @property
    def fullname(self):
        return self._module + '.' + self._name

    @property
    def available_plugins(self):
        return list(self.plugins.keys())

    def add_overload(self, sig, ck):
        """
        Adds a single signature and its ckernel to the overload resolver.
        """
        self.overloader.extend_overloads([sig])
        self.ckernels.append(ck)

    def add_plugin_overload(self, sig, data, pluginname):
        """
        Adds a single signature and corresponding data for a plugin
        implementation of the function.
        """
        # Get the overloader and data list for the plugin
        overloader, datalist = self.plugins.get(pluginname, (None, None))
        if overloader is None:
            overloader = datashape.OverloadResolver(self.fullname)
            datalist = []
            self.plugins[pluginname] = (overloader, datalist)
        # Add the overload
        overloader.extend_overloads([sig])
        datalist.append(data)

    def __call__(self, *args):
        """
        Apply blaze kernel `kernel` to the given arguments.

        Returns: a Deferred node representation the delayed computation
        """
        # Convert the arguments into blaze.Array
        args = [blaze.array(a) for a in args]

        # Merge input contexts
        ctxs = [term.expr[1] for term in args
                if isinstance(term, blaze.Array) and term.expr]
        ctx = ExprContext(ctxs)

        # Find match to overloaded function
        argstype = coretypes.Tuple([a.dshape for a in args])
        idx, match = self.overloader.resolve_overload(argstype)
        overload = Overload(match, self.overloader[idx], self.ckernels[idx])

        # Construct graph
        term = construct(self, ctx, overload, args)
        desc = DeferredDescriptor(term.dshape, (term, ctx))

        return blaze.Array(desc)

    def __str__(self):
        return "BlazeFunc %s" % self.name

    def __repr__(self):
        # TODO proper repr
        return str(self)


def _normalized_sig(sig):
    sig = datashape.dshape(sig)
    if len(sig) == 1:
        sig = sig[0]
    if not isinstance(sig, coretypes.Function):
        raise TypeError(('Only function signatures allowed as' +
                         'overloads, not %s') % sig)
    return sig


def _prepend_to_ds(ds, typevar):
    if isinstance(ds, coretypes.DataShape):
        tlist = ds.parameters
    else:
        tlist = (ds,)
    return coretypes.DataShape(typevar, *tlist)


def _add_elementwise_dims_to_sig(sig, typevarname):
    sig = _normalized_sig(sig)
    # Process the signature to add 'Dims... *' broadcasting
    if datashape.has_ellipsis(sig):
        raise TypeError(('Signature provided to ElementwiseBlazeFunc' +
                         'already includes ellipsis: %s') % sig)
    dims = coretypes.Ellipsis(coretypes.TypeVar(typevarname))
    params = [_prepend_to_ds(param, dims)
              for param in sig.parameters]
    return coretypes.Function(*params)


class ElementwiseBlazeFunc(BlazeFunc):
    """
    This is a kind of BlazeFunc that is always processed element-wise.
    When overloads are added to it, they have 'Dims... *' prepend
    the the datashape of every argument and the return type.
    """
    def add_overload(self, sig, ck):
        # Prepend 'Dims... *' to args and return type
        sig = _add_elementwise_dims_to_sig(sig, 'Dims')
        BlazeFunc.add_overload(self, sig, ck)

    def add_plugin_overload(self, sig, data, pluginname):
        # Prepend 'Dims... *' to args and return type
        sig = _add_elementwise_dims_to_sig(sig, 'Dims')
        BlazeFunc.add_plugin_overload(self, sig, data, pluginname)


class _ReductionResolver(object):
    """
    This is a helper class which resolves the output dimensions
    of a ReductionBlazeFunc call based on the 'axis=' and 'keepdims='
    arguments.
    """
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
        self.dimsin = coretypes.Ellipsis(coretypes.TypeVar('DimsIn'))
        self.dimsout = coretypes.Ellipsis(coretypes.TypeVar('DimsOut'))

    def __call__(self, sym, tvdict):
        if sym == self.dimsout:
            dims = tvdict[self.dimsin]
            # Create an array of flags indicating which dims are reduced
            if self.axis is None:
                dimflags = [True] * len(dims)
            else:
                dimflags = [False] * len(dims)
                try:
                    for ax in self.axis:
                        dimflags[ax] = True
                except IndexError:
                    raise IndexError(('axis %s is out of bounds for the' +
                                      'input type') % self.axis)
            # Remove or convert the reduced dims to fixed size-one
            if self.keepdims:
                reddim = coretypes.Fixed(1)
                return [reddim if dimflags[i] else dim
                        for i, dim in enumerate(dims)]
            else:
                return [dim for i, dim in enumerate(dims) if not dimflags[i]]


class ReductionBlazeFunc(BlazeFunc):
    """
    This is a kind of BlazeFunc with a calling convention for
    elementwise reductions which support 'axis=' and 'keepdims='
    keyword arguments.
    """
    def add_overload(self, sig, ck, associative, commutative, identity=None):
        sig = _normalized_sig(sig)
        if datashape.has_ellipsis(sig):
            raise TypeError(('Signature provided to ReductionBlazeFunc' +
                             'already includes ellipsis: %s') % sig)
        if len(sig.argtypes) != 1:
            raise TypeError(('Signature provided to ReductionBlazeFunc' +
                             'must have only one argument: %s') % sig)
        # Prepend 'DimsIn... *' to the args, and 'DimsOut... *' to
        # the return type
        sig = coretypes.Function(_prepend_to_ds(sig.argtypes[0],
                                                coretypes.Ellipsis(coretypes.TypeVar('DimsIn'))),
                                 _prepend_to_ds(sig.restype,
                                                coretypes.Ellipsis(coretypes.TypeVar('DimsOut'))))
        # TODO: This probably should be an object instead of a dict
        info = {'tag': 'reduction',
                'ckernel': ck,
                'assoc': associative,
                'comm': commutative,
                'ident': identity}
        BlazeFunc.add_overload(self, sig, info)

    def add_plugin_overload(self, sig, data, pluginname):
        raise NotImplementedError('TODO: implement add_plugin_overload')

    def __call__(self, *args, **kwargs):
        """
        Apply blaze kernel `kernel` to the given arguments.

        Returns: a Deferred node representation the delayed computation
        """
        # Validate the 'axis=' and 'keepdims=' keyword-only arguments
        axis = kwargs.pop('axis', None)
        if axis is not None and not isinstance(axis, tuple):
            axis = (axis,)
        keepdims = kwargs.pop('keepdims', False)
        if kwargs:
            msg = "%s got an unexpected keyword argument '%s'"
            raise TypeError(msg % (self.fullname, kwargs.keys()[0]))

        # Convert the arguments into blaze.Array
        args = [blaze.array(a) for a in args]

        # Merge input contexts
        ctxs = [term.expr[1] for term in args
                if isinstance(term, blaze.Array) and term.expr]
        ctx = ExprContext(ctxs)

        # Find match to overloaded function
        redresolver = _ReductionResolver(axis, keepdims)
        argstype = coretypes.Tuple([a.dshape for a in args])
        idx, match = self.overloader.resolve_overload(argstype, redresolver)
        info = dict(self.ckernels[idx])
        info['axis'] = axis
        info['keepdims'] = keepdims
        overload = Overload(match, self.overloader[idx], info)

        # Construct graph
        term = construct(self, ctx, overload, args)
        desc = DeferredDescriptor(term.dshape, (term, ctx))

        return blaze.Array(desc)
