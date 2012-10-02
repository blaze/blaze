"""
Blaze magic extensions for IPython.

Load with
In [0]: %load_ext ndtable.datashape.magic

In [1]: %%blaze
"""

from ndtable.datashape.parse import load, parse
from IPython.core.magic import Magics, magics_class, cell_magic

_loaded = False

@magics_class
class BlazeMagics(Magics):

    @cell_magic
    def blaze(self, line, cell):
        return parse(str(cell))

def load_ipython_extension(ip):
    global _loaded
    if not _loaded:
        ip.register_magics(BlazeMagics)
        _loaded = True
