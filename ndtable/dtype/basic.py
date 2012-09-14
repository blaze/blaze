import math
import ctypes
from collections import namedtuple

LOG2 = math.log(2)

_default_bytes = {
    'bool': 1,
    'int' : ctypes.sizeof(ctypes.c_char_p),
    'float' : ctypes.sizeof(ctypes.c_double),
    'complex': ctypes.sizeof(ctypes.c_double)*2
    }

def _set_and_order_fields(clsname, dct):
    # Loop through the dictionary and create a __pod__ named tuple that orders 
    #   any fields provided
    # If a __pod__ attribute exists, then extend it
    names = [(x._DType__counter, key) for key, x in dct.items() \
             if isinstance(x, DType)]
    # Sort the list of names by the counter
    names.sort(key=lambda x: x[0]) 
    prior = dct.get('__pod__', None)
    field_names = tuple(name for _, name in names)
    values = tuple(dct[name] for name in field_names)
    if prior is not None:
       pdict = prior._asdict()
       old_names, old_values = zip(*pdict.items())
       field_names = old_names + field_names
       values = old_values + values
    new = namedtuple(clsname+'_pod', field_names)
    dct['__pod__'] = new._make(values)
    # Remove the attributes moved into the pod
    for _, name in names:
        del dct[name]
    
class dtype(type):
    def __new__(cls, name, bases, dct):
        # Bypass if DType base-class being processed
        if name == 'DType':
            return type.__new__(cls, name, bases, dct)
        _set_and_order_fields(name, dct)
        return type.__new__(cls, name, bases, dct)

class DType(object):
    __metaclass__ = dtype
    __counter = 0
    def __init__(self):
        self.__counter = DType.__counter
        DType.__counter = self.__counter + 1
    def _validate_bits(self):
        try:
            valid = self._valid
        except AttributeError:
            return
        lower = valid.get('min', 8)
        upper = valid.get('max', 128)
        minpow = int(math.log(lower)/LOG2)
        maxpow = int(math.log(upper)/LOG2) + 1
        if self.bits not in [2**x for x in range(minpow, maxpow)]:
            message = "Invalid number of bits (%d) for type %s" % (self.bits,
                                                    self.__class__.__name__)
            raise TypeError(message)

class Basic(DType):
    def __init__(self, bits=None):
        super(Basic, self).__init__()
        if bits is None:
            bits = self._default
        self.bits = bits
        self._validate_bits()
        self.char = self._charmap.get(self.bits, '')
    
class Bool(Basic):
    _valid = {'min': 8, 'max': 128}
    _default = 8*_default_bytes['bool']
    _charmap = {8: '?'}
    kind = 'b'

class UInt(Basic):
    _valid = {'min': 8, 'max': 128}
    _default = 8*_default_bytes['int']
    _charmap = {8: 'B', 16: 'H', 32: 'I', 64: 'Q'}    
    kind = 'u'
    
class Int(Basic):
    _valid = {'min': 8, 'max': 128}
    _default = 8*_default_bytes['int']
    _charmap = {8: 'b', 16: 'h', 32: 'i', 64: 'q'}
    kind = 'i'

class Float(Basic):
    _valid = {'min': 16, 'max': 128}
    _default = 8*_default_bytes['float']
    _charmap = {16: 'e', 32: 'f', 64: 'd', 128: 'g'}
    kind = 'f'
    
class Complex(Basic):
    _valid = {'min': 32, 'max': 256}
    _default = 8*_default_bytes['complex']
    _charmap = {32: 'E', 64: 'F', 128: 'D', 256: 'G'}
    kind = 'c'
