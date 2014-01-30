"""Base classes for blaze distributed computing."""

from __future__ import absolute_import, division, print_function


class Distributed(object):
    """Abstract base class for all distributed objects.

    This class holds the communicator, serializer, and decomposition objects.
    """

    def __init__(self, comm, serializer, decomposition):
        self.comm = comm
        self.serializer = serializer
        self.decomposition = decomposition


class Communicator(object):
    """Abstract base class for all communicators"""

    pass


class Serializer(object):
    """Abstract base class for all serializers"""

    def dump(self, obj):
        """Dump an object to the serialization handle"""
        raise NotImplementedError("Base class should not be instantiated.")


