from textwrap import dedent
from blaze.cgen.blirgen import *

#------------------------------------------------------------------------
# Code Generation ( Level 1 )
#------------------------------------------------------------------------

# Level 1 just tests syntatic construction.

def test_compose():
    x = Assign('a', '3')
    y = Arg('int', 'a')
    z = VarDecl('int', 'x', '0')

    loop = For('x', Range('1', '2'), Block([x]))

    body = Block([z, loop])

    fn = FuncDef(
        name = 'kernel',
        args = [y],
        ret = 'void',
        body = body,
    )

    # XXX
    assert str(fn) is not None
