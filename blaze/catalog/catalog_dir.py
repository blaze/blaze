from __future__ import absolute_import, division, print_function

from os import path
import yaml
from .catalog_arr import load_blaze_array


def is_valid_bpath(d):
    """Returns true if it's a valid blaze path"""
    # Disallow backslashes in blaze paths
    if '\\' in d:
        return False
    # There should not be multiple path separators in a row
    if '//' in d:
        return False
    return True


def is_abs_bpath(d):
    """Returns true if it's an absolute blaze path"""
    return is_valid_bpath(d) and d.startswith('/')


def is_rel_bpath(d):
    """Returns true if it's a relative blaze path"""
    return is_valid_bpath(d) and not d.startswith('/')


def _clean_bpath_components(components):
    res = []
    for c in components:
        if c == '.':
            # Remove '.'
            pass
        elif c == '..':
            if all(x == '..' for x in res):
                # Relative path starting with '..'
                res.append('..')
            elif res == ['']:
                # Root of absolute path
                raise ValueError('Cannot use ".." at root of blaze catalog')
            else:
                # Remove the last entry
                res.pop()
        else:
            res.append(c)
    return res


def _split_bpath(d):
    if is_valid_bpath(d):
        if d == '':
            return []
        elif d == '/':
            return ['']
        elif d.endswith('/'):
            d = d[:-1]
        return d.split('/')
    else:
        raise ValueError('Invalid blaze catalog path %r' % d)


def _rejoin_bpath(components):
    if components == ['']:
        return '/'
    else:
        return '/'.join(components)


def clean_bpath(d):
    if is_valid_bpath(d):
        components = _split_bpath(d)
        components = _clean_bpath_components(components)
        return _rejoin_bpath(components)
    else:
        raise ValueError('Invalid blaze catalog path %r' % d)


def join_bpath(d1, d2):
    if is_abs_bpath(d2):
        return clean_bpath(d2)
    elif is_abs_bpath(d1):
        components = _split_bpath(d1) + _split_bpath(d2)
        components = _clean_bpath_components(components)
        return _rejoin_bpath(components)


class CatalogDir(object):
    """This object represents a directory path within the blaze catalog"""
    def __init__(self, conf, dir):
        self.conf = conf
        self.dir = dir
        if not is_abs_bpath(dir):
            raise ValueError('Require an absolute blaze path: %r' % dir)
        self._fsdir = path.join(conf.root, dir[1:])
        if not path.exists(self._fsdir) or not path.isdir(self._fsdir):
            raise RuntimeError('Blaze path not found: %r' % dir)

    def ls_arrs(self):
        """Return a list of all the arrays in this blaze dir"""
        return self.conf.ls_arrs(self.dir)

    def ls_dirs(self):
        """Return a list of all the directories in this blaze dir"""
        return self.conf.ls_dirs(self.dir)

    def ls(self):
        """
        Returns a list of all the arrays and directories in this blaze dir
        """
        return self.conf.ls(self.dir)

    def __getindex__(self, key):
        if isinstance(key, tuple):
            key = '/'.join(key)
        if not is_rel_bpath(key):
            raise ValueError('Require a relative blaze path: %r' % key)
        dir = '/'.join([self.dir, key])
        fsdir = path.join(self._fsdir, dir)
        if path.isdir(fsdir):
            return CatalogDir(self.conf, dir)
        elif path.isfile(fsdir + '.array'):
            return load_blaze_array(self.conf, dir)
        else:
            raise RuntimeError('Blaze path not found: %r' % dir)

    def __repr__(self):
        return ("Blaze Catalog Directory\nconfig: %s\ndir: %s"
                % (self.conf.configfile, self.dir))


class CatalogCDir(CatalogDir):
    """This object represents a directory path within a special catalog"""
    def __init__(self, conf, dir, subdir='/'):
        self.conf = conf
        self.dir = dir
        self.subdir = subdir
        if not is_abs_bpath(dir):
            raise ValueError('Require a path to dir file: %r' % dir)
        self._fsdir = path.join(conf.root, dir[1:])
        if not path.exists(self._fsdir + '.dir'):
            raise RuntimeError('Blaze path not found: %r' % dir)
        self.load_blaze_dir()

    def load_blaze_dir(self):
        fsdir = self.conf.get_fsdir(self.dir)
        with open(fsdir + '.dir') as f:
            dirmeta = yaml.load(f)
        self.ctype = dirmeta['type']
        imp = dirmeta['import']
        self.fname = imp.get('filename')

    def ls_arrs(self):
        """Return a list of all the arrays in this blaze dir"""
        if self.ctype == "hdf5":
            import tables as tb
            with tb.open_file(self.fname, 'r') as f:
                leafs = [l._v_name for l in
                         f.iter_nodes(self.subdir, classname='Leaf')]
            return sorted(leafs)

    def ls_dirs(self):
        """Return a list of all the directories in this blaze dir"""
        if self.ctype == "hdf5":
            import tables as tb
            with tb.open_file(self.fname, 'r') as f:
                groups = [g._v_name for g in 
                          f.iter_nodes(self.subdir, classname='Group')]
            return sorted(groups)

    def ls(self):
        """
        Returns a list of all the arrays and directories in this blaze dir
        """
        if self.ctype == "hdf5":
            import tables as tb
            with tb.open_file(self.fname, 'r') as f:
                nodes = [n._v_name for n in
                         f.iter_nodes(self.subdir)]
            return sorted(nodes)

    def ls_abs(self, cname=''):
        """
        Returns a list of all the directories in this blaze dir
        """
        if self.ctype == "hdf5":
            import tables as tb
            with tb.open_file(self.fname, 'r') as f:
                nodes = [n._v_pathname for n in
                         f.walk_nodes(self.subdir, classname=cname)]
            return sorted(nodes)

    def __getindex__(self, key):
        # XXX Adapt this to HDF5
        if isinstance(key, tuple):
            key = '/'.join(key)
        if not is_rel_bpath(key):
            raise ValueError('Require a relative blaze path: %r' % key)
        dir = '/'.join([self.dir, key])
        fsdir = path.join(self._fsdir, dir)
        if path.isfile(fsdir + '.dir'):
            return CatalogCDir(self.conf, dir)
        elif path.isfile(fsdir + '.array'):
            return load_blaze_array(self.conf, dir)
        else:
            raise RuntimeError('Blaze path not found: %r' % dir)
