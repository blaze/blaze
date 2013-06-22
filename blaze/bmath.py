#bmath.py
from blaze.blfuncs import BlazeFunc

_funcs = {'add':'+',
          'mul':'*',
          'sub':'-',
          'div':'/',
          'truediv':'/',
          'floordiv':'//',
          'mod': '%',
          'eq': '==',
          'ne': '!=',
          'lt': '<',
          'le': '<=',
          'gt': '>',
          'ge': '>='}

gl = globals()

template = """
def _{name}(a, b):
    return a {op} b
"""
for key, value in _funcs.items():
    exec(template.format(name=key, op=value))
    gl[key] = BlazeFunc(key, template=gl['_%s'%key])


