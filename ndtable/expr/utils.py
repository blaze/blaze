from string import letters
from itertools import count

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
