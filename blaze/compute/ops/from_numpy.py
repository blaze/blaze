"""
A helper function which turns a NumPy ufunc into a Blaze ufunc.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from dynd import _lowlevel
import datashape
from .. import function


def _filter_tplist(tplist):
    """Removes duplicates (arising from the long type usually), and
    eliminates the object dtype.
    """
    elim_kinds = ['O', 'M', 'm', 'S', 'U']
    if str(np.longdouble) != str(np.double):
        elim_types = [np.longdouble, np.clongdouble]
    else:
        elim_types = []
    elim_types.append(np.float16)
    seen = set()
    tplistnew = []
    for sig in tplist:
        if sig not in seen and not any(dt.kind in elim_kinds or
                                       dt in elim_types for dt in sig):
            tplistnew.append(sig)
            seen.add(sig)
    return tplistnew


def _make_sig(tplist):
    """Converts a type tuples into datashape function signatures"""
    dslist = [datashape.dshape("A... * " + str(x)) for x in tplist]
    return datashape.Function(*(dslist[1:] + [dslist[0]]))


def blazefunc_from_numpy_ufunc(uf, modname, name, acquires_gil):
    """Converts a NumPy ufunc into a Blaze ufunc.

    Parameters
    ----------
    uf : NumPy ufunc
        The ufunc to convert.
    modname : str
        The module name to report in the ufunc's name
    name : str
        The ufunc's name.
    acquires_gil : bool
        True if the kernels in the ufunc need the GIL.
        TODO: should support a dict {type -> bool} to allow per-kernel control.
    """
    # Get the list of type signatures
    tplist = _lowlevel.numpy_typetuples_from_ufunc(uf)
    tplist = _filter_tplist(tplist)
    siglist = [_make_sig(tp) for tp in tplist]
    kernlist = [_lowlevel.ckernel_deferred_from_ufunc(uf, tp, acquires_gil)
                for tp in tplist]
    # Create the empty blaze function to start
    blaze_func = function.BlazeFunc('blaze', name)
    blaze_func.add_metadata({'elementwise': True})
    # Add dispatching to the kernel for each signature
    ckdispatcher = blaze_func.get_dispatcher('ckernel')
    for (tp, sig, kern) in zip(tplist, siglist, kernlist):
        datashape.overloading.overload(sig, dispatcher=ckdispatcher)(kern)
    return blaze_func

