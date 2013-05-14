# -*- coding: utf-8 -*-
########################################################################
#
#       License: BSD
#       Created: August 16, 2012
#       Author:  Francesc Alted - francesc@continuum.io
#
########################################################################

from __future__ import absolute_import

import sys
import os, os.path
import json
from ..py3help import xrange, dict_iteritems

ATTRSDIR = "__attrs__"

class attrs(object):
    """Accessor for attributes in carray objects.

    This class behaves very similarly to a dictionary, and attributes
    can be appended in the typical way::

       attrs['myattr'] = value

    And can be retrieved similarly::

       value = attrs['myattr']

    Attributes can be removed with::

       del attrs['myattr']

    This class also honors the `__iter__` and `__len__` special
    functions.  Moreover, a `getall()` method returns all the
    attributes as a dictionary.

    CAVEAT: The values should be able to be serialized with JSON for
    persistence.

    """

    def __init__(self, rootdir, mode, _new=False):
        self.rootdir = rootdir
        self.mode = mode
        self.attrs = {}

        if self.rootdir:
            self.attrsfile = os.path.join(self.rootdir, ATTRSDIR)

        if self.rootdir:
            if _new:
                self._create()
            else:
                self._open()

    def _create(self):
        if self.mode != 'r':
            # Empty the underlying file
            with open(self.attrsfile, 'wb') as rfile:
                rfile.write(json.dumps({}, ensure_ascii=True).encode('ascii'))
                rfile.write(b"\n")

    def _open(self):
        if not os.path.isfile(self.attrsfile):
            if self.mode != 'r':
                # Create a new empty file
                with open(self.attrsfile, 'wb') as rfile:
                    rfile.write(b"\n")
        # Get the serialized attributes
        with open(self.attrsfile, 'rb') as rfile:
            try:
                data = json.loads(rfile.read().decode('ascii'))
            except:
                raise IOError(
                    "Attribute file is not readable")
        self.attrs = data

    def _update_meta(self):
        """Update attributes on-disk."""
        if not self.rootdir:
            return
        with open(self.attrsfile, 'wb') as rfile:
            rfile.write(json.dumps(self.attrs, ensure_ascii=True).encode('ascii'))
            rfile.write(b"\n")

    def getall(self):
        return self.attrs.copy()

    def __getitem__(self, name):
        return self.attrs[name]

    def __setitem__(self, name, carray):
        if self.rootdir and self.mode == 'r':
            raise IOError(
                "Cannot modify an attribute when in 'r'ead-only mode")
        self.attrs[name] = carray
        self._update_meta()

    def __delitem__(self, name):
        """Remove the `name` attribute."""
        if self.rootdir and self.mode == 'r':
            raise IOError(
                "Cannot remove an attribute when in 'r'ead-only mode")
        del self.attrs[name]
        self._update_meta()

    def __iter__(self):
        return dict_iteritems(self.attrs)

    def __len__(self):
        return len(self.attrs)

    def __str__(self):
        if len(self.attrs) == 0:
            return "*no attrs*"
        fullrepr = ""
        for name in self.attrs:
            fullrepr += "%s : %s" % (name, self.attrs[name])
        return fullrepr

    def __repr__(self):
        if len(self.attrs) == 0:
            return str(self)
        fullrepr = ""
        for name in self.attrs:
            fullrepr += "%s : %r\n" % (name, self.attrs[name])
        return fullrepr
