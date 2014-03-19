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
from datashape import coretypes

from ..datadescriptor import DeferredDescriptor
from .expr import construct, ExprContext

Overload = namedtuple('Overload', 'resolved_sig, sig, func')


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
    fullname : string
        The fully qualified name of the function.

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


def _add_elementwise_dims_to_ds(ds, typevar):
    if isinstance(ds, coretypes.DataShape):
        tlist = ds.parameters
    else:
        tlist = (ds,)
    return coretypes.DataShape(typevar, *tlist)

def _add_elementwise_dims_to_sig(sig, typevarname):
    # Process the signature to add 'Dims... *' broadcasting
    sig = datashape.dshape(sig)
    if len(sig) == 1:
        sig = sig[0]
    if not isinstance(sig, coretypes.Function):
        raise TypeError(('Only function signatures allowed as' +
                         'overloads, not %s') % sig)
    if datashape.has_ellipsis(sig):
        raise TypeError(('Signature provided to ElementwiseBlazeFunc' +
                         'already includes ellipsis: %s') % sig)
    dims = coretypes.Ellipsis(coretypes.TypeVar(typevarname))
    params = [_add_elementwise_dims_to_ds(param, dims)
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
