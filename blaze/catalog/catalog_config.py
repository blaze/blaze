from __future__ import absolute_import

import yaml
import os
from os import path
from .catalog_dir import is_abs_bpath

class CatalogConfig(object):
    """This object stores a catalog configuration.
    """
    def __init__(self, catconfigfile):
        try:
            catconfigfile = path.abspath(catconfigfile)
            self.configfile = catconfigfile
            with open(catconfigfile) as f:
                cfg = yaml.load(f)
            if not isinstance(cfg, dict):
                raise RuntimeError(('Blaze catalog config file "%s" is ' +
                        'not valid') % catconfigfile)
            self.root = cfg.pop('root')
            # Allow ~/...
            self.root = path.expanduser(self.root)
            # For paths that are not absolute, make them relative
            # to the config file, so a catalog + config can
            # be easily relocatable.
            if not path.isabs(self.root):
                self.root = path.join(path.dirname(catconfigfile), self.root)
            self.root = path.abspath(self.root)

            if not path.exists(self.root):
                raise RuntimeError(('Root Blaze catalog dir "%s" ' +
                        'from config file "%s" does not exist')
                        % (self.root, catconfigfile))

            if len(cfg) != 0:
                raise KeyError('Extra Blaze catalog config options: %s'
                        % cfg.keys())
        except KeyError as e:
            raise KeyError('Missing Blaze catalog config option: %s' % e)

    def get_fsdir(self, dir):
        """Return the filesystem path of the blaze catalog path"""
        if is_abs_bpath(dir):
            return path.join(self.root, dir[1:])
        else:
            raise ValueError('Expected absolute blaze catalog path: %r' % dir)

    def isarray(self, dir):
        """Check if a blaze catalog path points to an existing array"""
        if is_abs_bpath(dir):
            return path.isfile(path.join(self.root, dir[1:]) + '.array')
        else:
            raise ValueError('Expected absolute blaze catalog path: %r' % dir)

    def isdir(self, dir):
        """Check if a blaze catalog path points to an existing directory"""
        if is_abs_bpath(dir):
            return path.isdir(path.join(self.root, dir[1:]))
        else:
            raise ValueError('Expected absolute blaze catalog path: %r' % dir)

    def ls_arrs(self, dir):
        """Return a list of all the arrays in the provided blaze catalog dir"""
        if is_abs_bpath(dir):
            fsdir = path.join(self.root, dir[1:])
            listing = os.listdir(fsdir)
            return [path.splitext(x)[0] for x in listing
                if x.endswith('.array')]
        else:
            raise ValueError('Expected absolute blaze catalog path: %r' % dir)

    def ls_dirs(self, dir):
        """Return a list of all the directories in the provided blaze catalog dir"""
        if is_abs_bpath(dir):
            fsdir = path.join(self.root, dir[1:])
            listing = os.listdir(fsdir)
            return [x for x in listing
                if path.isdir(path.join(fsdir, x))]
        else:
            raise ValueError('Expected absolute blaze catalog path: %r' % dir)

    def ls(self, dir):
        """Return a list of all the arrays and directories in the provided
           blaze catalog dir
           """
        if is_abs_bpath(dir):
            fsdir = path.join(self.root, dir[1:])
            listing = os.listdir(fsdir)
            res = [path.splitext(x)[0] for x in listing
                if x.endswith('.array')]
            res += [x for x in listing
                if path.isdir(path.join(fsdir, x))]
            return res
        else:
            raise ValueError('Expected absolute blaze catalog path: %r' % dir)

    def __repr__(self):
        return ("Blaze Configuration\n%s\n\nroot: %s"
                % (self.configfile, self.root))

def load_default_config():
    dcf = path.expanduser('~/.blaze/catalog.yaml')
    if not path.exists(dcf):
        with open(dcf) as f:
            f.write("### Blaze Catalog Configuration File\n")
            f.write("root: Arrays\n")
        os.mkdir(path.expanduser('~/.blaze/Arrays'))
    return CatalogConfig(dcf)
