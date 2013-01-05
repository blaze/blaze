"""
Django style class declarations mapping onto Record datashape types.
"""

from numpy import dtype
from coretypes import Record, Type, DataShape
from parse import parse

#------------------------------------------------------------------------
# Record Declarations
#------------------------------------------------------------------------

# XXX: Do we really want this metaclass???
class DeclMeta(type):
    def __new__(meta, name, bases, namespace):
        abstract = namespace.pop('__abstract', False)
        # for tests
        dummy = namespace.pop('__dummy', False)

        if abstract or dummy:
            cls = type(name, bases, namespace)
        else:
            cls = type.__new__(meta, name, bases, namespace)
            cls.construct.im_func(cls, namespace)
            if hasattr(cls, 'fields'):
                # Side-effectful operation to register the alias
                # with the parser
                fields = cls.fields.items() + cls._derived.items()
                rcd = Record(fields)
                Type.register(name, rcd)
        return cls

class RecordDecl(object):
    """
    Record object, declared as class. Provied to the datashape parser
    through the metaclass.
    """
    __metaclass__ = DeclMeta

    def construct(cls, fields):
        cls.fields = {}
        cls._derived = {}
        for name, value in fields.items():
            if isinstance(value, DataShape):
                cls.add(name, value)

    @classmethod
    def add(cls, name, field):
        cls.fields[name] = field

    @classmethod
    def to_dtype(self):
        """
        Convert a record class definition into a structured array dtype
        """
        n = dtype([
            (label, ty.to_dtype()) for label, ty in
            self.fields.iteritems()
        ])
        return n

def derived(sig=None):
    def wrapper(fn):
        if sig is not None:
            return parse(sig)
        else:
            raise NotImplementedError
    return wrapper

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def dtype_from_dict(dct):
    """Convert a dictionary of Python types into a structured
    array dtype

    Parameters
    ----------
    uri : str
        Specifies the URI for the Blaze object.  It can be a regular file too.
    mode : the open mode (string)
        Specifies the mode in which the object is opened.  The supported
        values are:

          * 'r' for read-only
          * 'w' for emptying the previous underlying data
          * 'a' for allowing read/write on top of existing data

    Returns
    -------
        out : numpy.dtype instance

    Example

        >>> dtype_from_dict({'i': [1, 2, 3, 4], 'f': [4.0, 3.0, 2.0, 1.0]})
        dtype([('i', '<i8'), ('f', '<f8')])

    """
    return dtype([(k, dtype(type(v[0]))) for k,v in dct.iteritems()])
