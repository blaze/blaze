"""Blaze distributed array module"""

from __future__ import absolute_import, division, print_function

from .base import Distributed


class DistArray(Distributed):
    """A Distributed Array
    """

    # Class level defaults to override bulk behaviour
    DEFAULT_SERIALIZER = None
    DEFAULT_DECOMPOSITION = None

    def __init__(self, comm, serializer=None, decomposition=None):
        super(DistArray, self).__init__(comm,
                                        serializer if serializer else self.DEFAULT_SERIALIZER,
                                        decomposition if decomposition else self.DEFAULT_DECOMPOSITION)
