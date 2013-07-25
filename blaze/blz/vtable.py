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

from ..py2help import _inttypes, imap, xrange
_inttypes += (np.integer,)

# BLZ utilities
from . import utils, attrs, arrayprint
import os, os.path
from .btable import btable


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
        if os.path.isdir(btables):
            btables = self._get_dir(btables)
        self._open(btables)

    def _open(self, btables):
        self.btables = list(btables)
        # Check that all dtypes in btables are consistent
        self.dtype = dt = btables[0].dtype
        for bt in self.btables:
            if dt != bt.dtype:
                raise TypeError("dtypes are not consistent")
                
        self.sizes = [0] + [len(bt) for bt in btables]
        self.cumsizes = np.cumsum(self.sizes)
        self.len = self.cumsizes[-1]

    def _get_dir(self, fsdir):
        """Open a directory made of BLZ files"""
        blzs = [ os.path.join(fsdir, d) for d in os.listdir(fsdir)
                 if os.path.isdir(os.path.join(fsdir, d)) ]
        print "blzs:", blzs
        btables = [ btable(rootdir=d) for d in blzs ]
        return btables
        
    def __len__(self):
        return self.len

    def gettable_idx(self, idx, stop=False):
        if stop:
            idx = idx - 1
        itable = self.cumsizes.searchsorted(idx, side='right') - 1
        iidx = idx - self.cumsizes[itable]
        if stop:
            iidx += 1
        return itable, iidx
        
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
            ibt, idx = self.gettable_idx(key)
            return self.btables[ibt][idx]
        # Slices
        elif type(key) == slice:
            (start, stop, step) = key.start, key.stop, key.step
            if step and step <= 0 :
                raise NotImplementedError(
                    "step in slice can only be positive")

        # From now on, will only deal with [start:stop:step] slices

        # Get the corrected values for start, stop, step
        (start, stop, step) = slice(start, stop, step).indices(self.len)
        # Build a numpy container
        n = utils.get_len_of_range(start, stop, step=1)
        ra = np.empty(shape=(n,), dtype=self.dtype)

        # Fill it by iterating through all the arrays
        stable, sidx = self.gettable_idx(start)
        etable, eidx = self.gettable_idx(stop, stop=True)
        sstart = sidx
        if etable == stable:
            sstop = eidx
        else:
            sstop = len(self.btables[stable])
        sout, eout = 0, sstop - sstart
        for i, bt in enumerate(self.btables[stable:etable+1]):
            ra[sout:eout] = bt[sstart:sstop]
            sout = eout
            sstart = 0
            if stable + i + 1 == etable:
                sstop = eidx 
            elif (stable + i + 1) < len(self.btables):
                sstop = len(self.btables[stable + i + 1])
            eout += sstop - sstart
        if step > 1:
            ra = ra[::step]
        return ra
