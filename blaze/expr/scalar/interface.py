from ..core import Expr
from .core import Scalar
from .numbers import NumberInterface
from .boolean import BooleanInterface
from datashape import dshape

class ScalarSymbol(NumberInterface, BooleanInterface):
    __slots__ = 'name', 'dtype'

    def __init__(self, name, dtype='real'):
        self.name = name
        self.dtype = dtype

    @property
    def dshape(self):
        return dshape(self.dtype)

    def __str__(self):
        return str(self.name)

    __hash__ = Expr.__hash__
