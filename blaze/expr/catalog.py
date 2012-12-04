#------------------------------------------------------------------------
# Method Maps
#------------------------------------------------------------------------

PyObject_BinaryOperators = [
    ('or','|'),  ('and','&'), ('xor','^'), ('lshift','<<'), ('rshift','>>'),
    ('add','+'), ('sub','-'), ('mul','*'), ('div','/'), ('pow','**'),
    ('mod','%'), ('truediv','/'), ('floordiv','//'), ('lt','<'), ('gt','>'),
    ('le','<='), ('ge','>='), ('eq','=='), ('ne','!=')
]

PyObject_UnaryOperators = [
    ('neg','-'), ('pos','+'), ('invert','~')
]

PyObject_Intrinsics = [
    'abs', 'complex', 'int', 'long', 'float', 'oct', 'hex'
    'cmp',
]

PyObject_Representation = [
    'str', 'repr'
]

PyArray_Intrinsics = [
    "dtype", "size"
]

PyArray_WriteMethods = [
    'fill', 'itemset', 'put', 'setflags', 'setfield'
]

PyArray_ReadMethods = [
    'all', 'any', 'argmax', 'argmin', 'argsort', 'astype', 'base', 'byteswap',
    'choose', 'clip', 'compress', 'conj', 'conjugate', 'copy', 'ctypes',
    'cumprod', 'cumsum', 'diagonal', 'dot', 'dtype', 'dump', 'dumps',
    'flatten', 'getfield', 'item', 'max', 'mean', 'min', 'nbytes',
    'newbyteorder', 'nonzero', 'prod', 'ptp', 'ravel', 'repeat', 'reshape',
    'resize', 'round', 'searchsorted', 'setasflat', 'sort', 'squeeze', 'std',
    'sum', 'swapaxes', 'take', 'trace', 'transpose', 'var', 'view'
]
