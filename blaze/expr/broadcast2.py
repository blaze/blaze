from datashape import *

from .expressions import *
from .arithmetic import maxshape

class Broadcast(Expr):
    __slots__ = '_children', '_scalars', '_scalar_expr'

    @property
    def dshape(self):
        myshape = maxshape(map(shape, self._children))
        return DataShape(*(myshape + (self._scalar_expr.schema,)))


