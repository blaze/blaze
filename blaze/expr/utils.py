from string import letters
from itertools import count

#------------------------------------------------------------------------
# Symbols
#------------------------------------------------------------------------

class Symbol:
    def __init__(self, name):
        self.__name = name
        self.__hash = hash(self.__name)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.__name == other.__name

    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            raise True
        return self.__name != other.__name

    # We require that these be unique in sets and dicts, ergo
    # same hash.
    def __hash__(self):
        return self.__hash

    def __repr__(self):
        return self.__name

#------------------------------------------------------------------------
# Free Variables
#------------------------------------------------------------------------

def _var_generator(prefix=None):
    """
    Generate a stream of unique free variables.
    """
    for a in count(0):
        for b in letters:
            if a == 0:
                yield (prefix or '') + b
            else:
                yield (prefix or '') + ''.join([str(a),str(b)])
