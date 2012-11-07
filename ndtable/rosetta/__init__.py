from parse import Rosetta, parse

def _pytables():
    import tables as pytables

    tr = Rosetta()
    tr.namespace = pytables

    expr = open('rosetta/pytables.table').read()
    stone = tr.visit(parse(expr))

    return dict(a.astuple() for a in stone)

try:
    pytables = _pytables()
except IOError:
    pytables = None
