"""
Helper functions which constructs blaze functions from dynd kernels.
"""

from __future__ import absolute_import, division, print_function

from dynd import _lowlevel
import datashape

from .. import function


def _make_sig(kern):
    dslist = [datashape.dshape("A..., " + str(x)) for x in kern.types]
    return datashape.Function(*(dslist[1:] + [dslist[0]]))


def _make_pyfunc(nargs, modname, name):
    if nargs == 1:
        def pyfunc(arg1):
            raise NotImplementedError('pyfunc for blaze func %s should not be called' % name)
    elif nargs == 2:
        def pyfunc(arg1, arg2):
            raise NotImplementedError('pyfunc for blaze func %s should not be called' % name)
    elif nargs == 3:
        def pyfunc(arg1, arg2, arg3):
            raise NotImplementedError('pyfunc for blaze func %s should not be called' % name)
    elif nargs == 4:
        def pyfunc(arg1, arg2, arg3, arg4):
            raise NotImplementedError('pyfunc for blaze func %s should not be called' % name)
    else:
        raise ValueError('unsupported number of args %s' % nargs)
    pyfunc.__module__ = modname
    pyfunc.__name__ = modname + '.' + name if modname else name
    return pyfunc


def blazefunc_from_dynd_property(tplist, propname, modname, name):
    """Converts a dynd property access into a Blaze ufunc.

    Parameters
    ----------
    tplist : list of dynd types
        A list of the types to use.
    propname : str
        The name of the property to access on the type.
    modname : str
        The module name to report in the ufunc's name
    name : str
        The ufunc's name.
    """
    # Get the list of type signatures
    kernlist = [_lowlevel.make_ckernel_deferred_from_property(tp, propname,
                                                              'expr', 'default')
                for tp in tplist]
    siglist = [_make_sig(kern) for kern in kernlist]
    # Create the empty blaze function to start
    blaze_func = function.BlazeFunc()
    blaze_func.add_metadata({'elementwise': True})
    # Add default dummy dispatching for 'python' mode
    pydispatcher = blaze_func.get_dispatcher('python')
    # Add dispatching to the kernel for each signature
    ckdispatcher = blaze_func.get_dispatcher('ckernel')
    for (sig, kern) in zip(siglist, kernlist):
        # The blaze function currently requires matching (do-nothing)
        # python functions
        pyfunc = _make_pyfunc(len(kern.types) - 1, modname, name)
        datashape.overloading.overload(sig, dispatcher=pydispatcher)(pyfunc)
        datashape.overloading.overload(sig, dispatcher=ckdispatcher)(kern)
    return blaze_func
