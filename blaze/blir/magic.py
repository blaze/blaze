"""
BLIR magic extensions for IPython.

Load with
In [0]: %load_ext blir.magic

In [1]: %%blir
"""

from IPython.core.magic import Magics, magics_class, cell_magic

_loaded = False

@magics_class
class BlazeMagics(Magics):

    @staticmethod
    def err_redirect(msg):
        print(msg)

    @cell_magic
    def blir(self, line, cell):
        from blir import compile, CompileError
        from blir.errors import listen

        try:
            with listen(self.err_redirect):
                ast, env = compile(str(cell))
            llvm = env['cgen'].module

            print(' LLVM '.center(80, '='))
            print(str(llvm))
        except CompileError as e:
            print('FAIL: Failure in compiler phase:', e.args[0])

def load_ipython_extension(ip):
    global _loaded
    if not _loaded:
        ip.register_magics(BlazeMagics)
        _loaded = True
