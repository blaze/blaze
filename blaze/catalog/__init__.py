from __future__ import absolute_import

from . import blaze_url
from .catalog_config import CatalogConfig, load_default_config
from .catalog_dir import CatalogDir, is_rel_bpath, is_abs_bpath, join_bpath
from .catalog_arr import load_blaze_array

# Load the default config (creates one if none exists)
config = load_default_config()
_cwd = '/'

def load_config(cfgfile):
    """Loads a fresh catalog from the specified config file"""
    global config, _cwd
    config = CatalogConfig(cfgfile)
    _cwd = '/'

def cd(key):
    """Change the current working directory of the blaze catalog"""
    global _cwd
    _cwd = join_bpath(_cwd, key)

def cwd():
    """The current working directory of the blaze catalog"""
    return _cwd

def ls_dirs(dir=None):
    """A list of the directories in the current working directory"""
    return config.ls_dirs(join_bpath(_cwd, dir) if dir else _cwd)

def ls_arrs(dir=None):
    """A list of the arrays in the current working directory"""
    return config.ls_arrs(join_bpath(_cwd, dir) if dir else _cwd)

def ls(dir=None):
    """A list of the arrays and directories in the current working directory"""
    return config.ls(join_bpath(_cwd, dir) if dir else _cwd)

def get(key):
    """Get an array or directory object from the blaze catalog"""
    key = join_bpath(_cwd, key)
    if config.isdir(key):
        return CatalogDir(config, key)
    elif config.isarray(key):
        return load_blaze_array(config, key)
    else:
        raise RuntimeError('Blaze path not found: %r' % key)

def register_ipy_magic():
    """Register some IPython line magic for the blaze catalog.

        Examples
        --------

        In [1]: import blaze

        In [2]: bls
        dates_vals kiva random

        In [3]: bls kiva
        lenders loans

        In [4]: bcd kiva
        /kiva

        In [5]: bpwd
        Out[5]: u'/kiva'

        In [6]: a = %barr loans

    """
    import IPython
    from IPython.core.magic import register_line_magic
    @register_line_magic
    def bcd(line):
        """Change the current Blaze catalog working directory"""
        cd(line)
        print(_cwd)
    @register_line_magic
    def bpwd(line):
        """Print/return the current Blaze catalog working directory"""
        return _cwd
    @register_line_magic
    def bls(line):
        """List the directories and arrays in the specified
        Blaze catalog directory, or the working directory if
        not specified."""
        ip = IPython.get_ipython()
        cs = ip.prompt_manager.color_scheme_table[ip.colors]
        dcol = cs.colors['in_number']
        ncol = cs.colors['normal']
        tmp = [(x, x) for x in ls_arrs(line)]
        tmp += [(x, dcol + x + ncol) for x in ls_dirs(line)]
        if tmp:
            print(' '.join(x[1] for x in sorted(tmp)))
    @register_line_magic
    def barr(line):
        """Returns the Blaze array from the catalog at the specified path"""
        key = join_bpath(_cwd, line)
        if config.isarray(key):
            return load_blaze_array(config, key)
        else:
            raise RuntimeError('Blaze path not found: %r' % key)
