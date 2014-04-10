"""
Deferred data descriptor for deferred expressions. This is backed up by an
actual deferred expression graph.
"""

from __future__ import absolute_import, division, print_function

import blaze
import datashape

from . import DDesc, Capabilities
from .dynd_data_descriptor import DyND_DDesc
from dynd import nd

#------------------------------------------------------------------------
# Data Descriptor
#------------------------------------------------------------------------

class Stream_DDesc(DDesc):
    """
    Data descriptor for arrays exposing mainly an iterator interface.

    Attributes:
    -----------
    dshape: datashape.dshape
        The datashape of the stream data descriptor.

    condition: string
        The condtion over the original array, in string form.
    """

    def __init__(self, iterator, dshape, condition):
        self._iterator = iterator
        # The length of the iterator is unknown, so we put 'var' here
        self._dshape = datashape.dshape("var * " + str(dshape.measure))
        #
        self.condition = condition

    @property
    def dshape(self):
        return self._dshape

    @property
    def capabilities(self):
        """The capabilities for the deferred data descriptor."""
        return Capabilities(
            immutable = True,
            deferred = False,
            stream = True,
            # persistency is not supported yet
            persistent = False,
            appendable = False,
            remote = False,
            )

    def __getitem__(self, key):
        """Streams do not support random seeks.
        """
        raise NotImplementedError

    def __iter__(self):
        return self._iterator

    def _printer(self):
        return "<Array(iter('%s'), '%s')>" % (self.condition, self.dshape,)

    def _printer_repr(self):
        return self._printer()
