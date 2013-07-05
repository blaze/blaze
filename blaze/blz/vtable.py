########################################################################
#
#       License: BSD
#       Created: July 05, 2013
#       Author:  Francesc Alted - francesc@continuum.io
#
########################################################################

"""
vtable stands for 'virtual table' and it is a way to group different
btables in a single, virtual one.
"""

from __future__ import absolute_import
import numpy as np

from ..py3help import _inttypes, imap, xrange

_inttypes += (np.integer,)

class vtable(object):
    """
    vtable(btables, rootdir=None)

    This class represents a collection of column-wise btable objects.

    Create a new vtable from `btables` being a list of btables that
    share the same dtype (this limitation could be removed in the
    future).  The vtable does not replicate the actual data, but
    rather is another layer of metadata on top of actual btables.

    Parameters
    ----------

    btables : tuple or list of btable objects
        The list of btables to build the vtable object.

    rootdir : string
        This is the directory name where the vtable will be stored for
        persistence.  If not specified, then the vtable will be
        ephemeral (i.e. in-memory).

    """

    def __init__(self, btables, rootdir=None):
        if rootdir != None:
            raise ValueError("Persistent vtable are not yet supported")

        self.btables = list(btables)
        # Check that all dtypes in btables are consistent
        for i, bt in enumerate(self.btables):
            if i == 0:
                dt = bt.dtype
            else:
                if dt != bt.dtype:
                    raise TypeError("dtypes are not consistent")
                
        self.sizes = [len(bt) for bt in btables]
        self.cumsizes = np.cumsum(self.sizes)

    def gettable_idx(self, idx):
        itable = self.cumsizes.searchsorted(idx)
        iidx = idx - self.cumsizes[itable]
        return self.btables[itable], iidx
        
    def __getitem__(self, key):
        """
        x.__getitem__(key) <==> x[key]

        Returns values based on `key`.  All the functionality of
        ``ndarray.__getitem__()`` is supported (including fancy
        indexing), plus a special support for expressions:

        Parameters
        ----------
        key : string, int, tuple, list
            The corresponding btable column name will be returned.  If not a
            column name, it will be interpret as a boolean expression
            (computed via `btable.eval`) and the rows where these values are
            true will be returned as a NumPy structured array.  If `key` is an
            integer, slice or list then the typical NumPy indexing operation
            will be performed over the table.

        """        

        # First, check for integer
        if isinstance(key, _inttypes):
            # Get the index for the btable
            bt, idx = self.gettable_idx(key)
            return bt[idx]
        # # Slices
        # elif type(key) == slice:
        #     (start, stop, step) = key.start, key.stop, key.step
        #     if step and step <= 0 :
        #         raise NotImplementedError(
        #             "step in slice can only be positive")
