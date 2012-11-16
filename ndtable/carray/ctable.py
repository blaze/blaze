########################################################################
#
#       License: BSD
#       Created: September 01, 2010
#       Author:  Francesc Alted - francesc@continuum.io
#
########################################################################

import sys, math

import numpy as np
import carray as ca
from carray import utils, attrs, array2string
import itertools as it
from collections import namedtuple
import json
import os, os.path
import shutil


ROOTDIRS = '__rootdirs__'

class cols(object):
    """Class for accessing the columns on the ctable object."""

    def __init__(self, rootdir, mode):
        self.rootdir = rootdir
        self.mode = mode
        self.names = []
        self._cols = {}

    def read_meta_and_open(self):
        """Read the meta-information and initialize structures."""
        # Get the directories of the columns
        rootsfile = os.path.join(self.rootdir, ROOTDIRS)
        with open(rootsfile, 'rb') as rfile:
            data = json.loads(rfile.read())
        # JSON returns unicode (?)
        self.names = [str(name) for name in data['names']]
        # Initialize the cols by instatiating the carrays
        for name, dir_ in data['dirs'].items():
            self._cols[str(name)] = ca.carray(rootdir=dir_, mode=self.mode)

    def update_meta(self):
        """Update metainfo about directories on-disk."""
        if not self.rootdir:
            return
        dirs = dict((n, o.rootdir) for n,o in self._cols.items())
        data = {'names': self.names, 'dirs': dirs}
        rootsfile = os.path.join(self.rootdir, ROOTDIRS)
        with open(rootsfile, 'wb') as rfile:
            rfile.write(json.dumps(data))
            rfile.write("\n")

    def __getitem__(self, name):
        return self._cols[name]

    def __setitem__(self, name, carray):
        self.names.append(name)
        self._cols[name] = carray
        self.update_meta()

    def __iter__(self):
        return iter(self._cols)

    def __len__(self):
        return len(self.names)

    def insert(self, name, pos, carray):
        """Insert carray in the specified pos and name."""
        self.names.insert(pos, name)
        self._cols[name] = carray
        self.update_meta()

    def pop(self, name):
        """Return the named column and remove it."""
        pos = self.names.index(name)
        name = self.names.pop(pos)
        col = self._cols[name]
        self.update_meta()
        return col
    
    def __str__(self):
        fullrepr = ""
        for name in self.names:
            fullrepr += "%s : %s" % (name, str(self._cols[name])) 
        return fullrepr

    def __repr__(self):
        fullrepr = ""
        for name in self.names:
            fullrepr += "%s : %s\n" % (name, repr(self._cols[name])) 
        return fullrepr


class ctable(object):
    """
    ctable(cols, names=None, **kwargs)

    This class represents a compressed, column-wise, in-memory table.

    Create a new ctable from `cols` with optional `names`.

    Parameters
    ----------
    columns : tuple or list of column objects
        The list of column data to build the ctable object.  This can also be
        a pure NumPy structured array.  A list of lists or tuples is valid
        too, as long as they can be converted into carray objects.
    names : list of strings or string
        The list of names for the columns.  The names in this list must be
        valid Python identifiers, must not start with an underscore, and has
        to be specified in the same order as the `cols`.  If not passed, the
        names will be chosen as 'f0' for the first column, 'f1' for the second
        and so on so forth (NumPy convention).
    kwargs : list of parameters or dictionary
        Allows to pass additional arguments supported by carray
        constructors in case new carrays need to be built.

    Notes
    -----
    Columns passed as carrays are not be copied, so their settings
    will stay the same, even if you pass additional arguments (cparams,
    chunklen...).

    """

    # Properties
    # ``````````

    @property
    def cbytes(self):
        "The compressed size of this object (in bytes)."
        return self._get_stats()[1]

    @property
    def cparams(self):
        "The compression parameters for this object."
        return self._cparams

    @property
    def dtype(self):
        "The data type of this object (numpy dtype)."
        names, cols = self.names, self.cols
        l = [(name, cols[name].dtype) for name in names]
        return np.dtype(l)

    @property
    def names(self):
        "The names of the object (list)."
        return self.cols.names

    @property
    def ndim(self):
        "The number of dimensions of this object."
        return len(self.shape)

    @property
    def nbytes(self):
        "The original (uncompressed) size of this object (in bytes)."
        return self._get_stats()[0]

    @property
    def shape(self):
        "The shape of this object."
        return (self.len,)

    @property
    def size(self):
        "The size of this object."
        return np.prod(self.shape)


    def __init__(self, columns=None, names=None, **kwargs):

        # Important optional params
        self._cparams = kwargs.get('cparams', ca.cparams())
        self.rootdir = kwargs.get('rootdir', None)
        "The directory where this object is saved."
        self.mode = kwargs.get('mode', 'a')
        "The mode in which the object is created/opened."
        
        # Setup the columns accessor
        self.cols = cols(self.rootdir, self.mode)
        "The ctable columns accessor."

        # The length counter of this array
        self.len = 0

        # Create a new ctable or open it from disk
        if columns is not None:
            self.create_ctable(columns, names, **kwargs)
            _new = True
        else:
            self.open_ctable()
            _new = False

        # Attach the attrs to this object
        self.attrs = attrs.attrs(self.rootdir, self.mode, _new=_new)
            
        # Cache a structured array of len 1 for ctable[int] acceleration
        self._arr1 = np.empty(shape=(1,), dtype=self.dtype)

    def create_ctable(self, columns, names, **kwargs):
        """Create a ctable anew."""

        # Create the rootdir if necessary
        if self.rootdir:
            self.mkdir_rootdir(self.rootdir, self.mode)

        # Get the names of the columns
        if names is None:
            if isinstance(columns, np.ndarray):  # ratype case
                names = list(columns.dtype.names)
            else:
                names = ["f%d"%i for i in range(len(columns))]
        else:
            if type(names) == tuple:
                names = list(names)
            if type(names) != list:
                raise ValueError(
                    "`names` can only be a list or tuple")
            if len(names) != len(columns):
                raise ValueError(
                    "`columns` and `names` must have the same length")
        # Check names validity
        nt = namedtuple('_nt', names, verbose=False)
        names = list(nt._fields)

        # Guess the kind of columns input
        calist, nalist, ratype = False, False, False
        if type(columns) in (tuple, list):
            calist = [type(v) for v in columns] == [ca.carray for v in columns]
            nalist = [type(v) for v in columns] == [np.ndarray for v in columns]
        elif isinstance(columns, np.ndarray):
            ratype = hasattr(columns.dtype, "names")
            if ratype:
                if len(columns.shape) != 1:
                    raise ValueError, "only unidimensional shapes supported"
        else:
            raise ValueError, "`columns` input is not supported"
        if not (calist or nalist or ratype):
            # Try to convert the elements to carrays
            try:
                columns = [ca.carray(col) for col in columns]
                calist = True
            except:
                raise ValueError, "`columns` input is not supported"

        # Populate the columns
        clen = -1
        for i, name in enumerate(names):
            if self.rootdir:
                # Put every carray under each own `name` subdirectory
                kwargs['rootdir'] = os.path.join(self.rootdir, name)
            if calist:
                column = columns[i]
                if self.rootdir:
                    # Store this in destination
                    column = column.copy(**kwargs)
            elif nalist:
                column = columns[i]
                if column.dtype == np.void:
                    raise ValueError,(
                        "`columns` elements cannot be of type void")
                column = ca.carray(column, **kwargs)
            elif ratype:
                column = ca.carray(columns[name], **kwargs)
            self.cols[name] = column
            if clen >= 0 and clen != len(column):
                raise ValueError, "all `columns` must have the same length"
            clen = len(column)
 
        self.len = clen

    def open_ctable(self):
        """Open an existing ctable on-disk."""

        if self.rootdir is None:
            raise ValueError(
                "you need to pass either a `columns` or a `rootdir` param")

        # Open the ctable by reading the metadata
        self.cols.read_meta_and_open()

        # Get the length out of the first column
        self.len = len(self.cols[self.names[0]])

    def mkdir_rootdir(self, rootdir, mode):
        """Create the `self.rootdir` directory safely."""
        if os.path.exists(rootdir):
            if mode != "w":
                raise RuntimeError(
                    "specified rootdir path '%s' already exists "
                    "and creation mode is '%s'" % (rootdir, mode))
            if os.path.isdir(rootdir):
                shutil.rmtree(rootdir)
            else:
                os.remove(rootdir)
        os.mkdir(rootdir)

    def append(self, rows):
        """
        append(rows)

        Append `rows` to this ctable.

        Parameters
        ----------
        rows : list/tuple of scalar values, NumPy arrays or carrays
            It also can be a NumPy record, a NumPy recarray, or
            another ctable.

        """

        # Guess the kind of rows input
        calist, nalist, sclist, ratype = False, False, False, False
        if type(rows) in (tuple, list):
            calist = [type(v) for v in rows] == [ca.carray for v in rows]
            nalist = [type(v) for v in rows] == [np.ndarray for v in rows]
            if not (calist or nalist):
                # Try with a scalar list
                sclist = True
        elif isinstance(rows, np.ndarray):
            ratype = hasattr(rows.dtype, "names")
        elif isinstance(rows, ca.ctable):
            # Convert int a list of carrays
            rows = [rows[name] for name in self.names]
            calist = True
        else:
            raise ValueError, "`rows` input is not supported"
        if not (calist or nalist or sclist or ratype):
            raise ValueError, "`rows` input is not supported"

        # Populate the columns
        clen = -1
        for i, name in enumerate(self.names):
            if calist or sclist:
                column = rows[i]
            elif nalist:
                column = rows[i]
                if column.dtype == np.void:
                    raise ValueError, "`rows` elements cannot be of type void"
                column = column
            elif ratype:
                column = rows[name]
            # Append the values to column
            self.cols[name].append(column)
            if sclist:
                clen2 = 1
            else:
                clen2 = len(column)
            if clen >= 0 and clen != clen2:
                raise ValueError, "all cols in `rows` must have the same length"
            clen = clen2
        self.len += clen

    def trim(self, nitems):
        """
        trim(nitems)

        Remove the trailing `nitems` from this instance.

        Parameters
        ----------
        nitems : int
            The number of trailing items to be trimmed.

        """

        for name in self.names:
            self.cols[name].trim(nitems)
        self.len -= nitems

    def resize(self, nitems):
        """
        resize(nitems)

        Resize the instance to have `nitems`.

        Parameters
        ----------
        nitems : int
            The final length of the instance.  If `nitems` is larger than the
            actual length, new items will appended using `self.dflt` as
            filling values.

        """

        for name in self.names:
            self.cols[name].resize(nitems)
        self.len = nitems

    def addcol(self, newcol, name=None, pos=None, **kwargs):
        """
        addcol(newcol, name=None, pos=None, **kwargs)

        Add a new `newcol` object as column.

        Parameters
        ----------
        newcol : carray, ndarray, list or tuple
            If a carray is passed, no conversion will be carried out.
            If conversion to a carray has to be done, `kwargs` will
            apply.
        name : string, optional
            The name for the new column.  If not passed, it will
            receive an automatic name.
        pos : int, optional
            The column position.  If not passed, it will be appended
            at the end.
        kwargs : list of parameters or dictionary
            Any parameter supported by the carray constructor.

        Notes
        -----
        You should not specificy both `name` and `pos` arguments,
        unless they are compatible.

        See Also
        --------
        delcol

        """

        # Check params
        if pos is None:
            pos = len(self.names)
        else:
            if pos and type(pos) != int:
                raise ValueError, "`pos` must be an int"
            if pos < 0 or pos > len(self.names):
                raise ValueError, "`pos` must be >= 0 and <= len(self.cols)"
        if name is None:
            name = "f%d" % pos
        else:
            if type(name) != str:
                raise ValueError, "`name` must be a string"
        if name in self.names:
            raise ValueError, "'%s' column already exists" % name
        if len(newcol) != self.len:
            raise ValueError, "`newcol` must have the same length than ctable"

        if isinstance(newcol, np.ndarray):
            if 'cparams' not in kwargs:
                kwargs['cparams'] = self.cparams
            newcol = ca.carray(newcol, **kwargs)
        elif type(newcol) in (list, tuple):
            if 'cparams' not in kwargs:
                kwargs['cparams'] = self.cparams
            newcol = ca.carray(newcol, **kwargs)
        elif type(newcol) != ca.carray:
            raise ValueError(
                """`newcol` type not supported""")

        # Insert the column
        self.cols.insert(name, pos, newcol)
        # Update _arr1
        self._arr1 = np.empty(shape=(1,), dtype=self.dtype)

    def delcol(self, name=None, pos=None):
        """
        delcol(name=None, pos=None)

        Remove the column named `name` or in position `pos`.

        Parameters
        ----------
        name: string, optional
            The name of the column to remove.
        pos: int, optional
            The position of the column to remove.

        Notes
        -----
        You must specify at least a `name` or a `pos`.  You should not
        specify both `name` and `pos` arguments, unless they are
        compatible.

        See Also
        --------
        addcol

        """

        if name is None and pos is None:
            raise ValueError, "specify either a `name` or a `pos`"
        if name is not None and pos is not None:
            raise ValueError, "you cannot specify both a `name` and a `pos`"
        if name:
            if type(name) != str:
                raise ValueError, "`name` must be a string"
            if name not in self.names:
                raise ValueError, "`name` not found in columns"
            pos = self.names.index(name)
        elif pos is not None:
            if type(pos) != int:
                raise ValueError, "`pos` must be an int"
            if pos < 0 or pos > len(self.names):
                raise ValueError, "`pos` must be >= 0 and <= len(self.cols)"
            name = self.names[pos]

        # Remove the column
        self.cols.pop(name)
        # Update _arr1
        self._arr1 = np.empty(shape=(1,), dtype=self.dtype)

    def copy(self, **kwargs):
        """
        copy(**kwargs)

        Return a copy of this ctable.

        Parameters
        ----------
        kwargs : list of parameters or dictionary
            Any parameter supported by the carray/ctable constructor.

        Returns
        -------
        out : ctable object
            The copy of this ctable.

        """

        # Check that origin and destination do not overlap
        rootdir = kwargs.get('rootdir', None)
        if rootdir and self.rootdir and  rootdir == self.rootdir:
                raise RuntimeError("rootdir cannot be the same during copies")

        # Remove possible unsupported args for columns
        names = kwargs.pop('names', self.names)

        # Copy the columns
        if rootdir:
            # A copy is always made during creation with a rootdir
            cols = [ self.cols[name] for name in self.names ]
        else:
            cols = [ self.cols[name].copy(**kwargs) for name in self.names ]
        # Create the ctable
        ccopy = ctable(cols, names, **kwargs)
        return ccopy

    def __len__(self):
        return self.len

    def __sizeof__(self):
        return self.cbytes

    def where(self, expression, outcols=None, limit=None, skip=0):
        """
        where(expression, outcols=None, limit=None, skip=0)

        Iterate over rows where `expression` is true.

        Parameters
        ----------
        expression : string or carray
            A boolean Numexpr expression or a boolean carray.
        outcols : list of strings or string
            The list of column names that you want to get back in results.
            Alternatively, it can be specified as a string such as 'f0 f1' or
            'f0, f1'.  If None, all the columns are returned.  If the special
            name 'nrow__' is present, the number of row will be included in
            output.
        limit : int
            A maximum number of elements to return.  The default is return
            everything.
        skip : int
            An initial number of elements to skip.  The default is 0.

        Returns
        -------
        out : iterable
            This iterable returns rows as NumPy structured types (i.e. they
            support being mapped either by position or by name).

        See Also
        --------
        iter

        """

        # Check input
        if type(expression) is str:
            # That must be an expression
            boolarr = self.eval(expression)
        elif hasattr(expression, "dtype") and expression.dtype.kind == 'b':
            boolarr = expression
        else:
            raise ValueError, "only boolean expressions or arrays are supported"

        # Check outcols
        if outcols is None:
            outcols = self.names
        else:
            if type(outcols) not in (list, tuple, str):
                raise ValueError, "only list/str is supported for outcols"
            # Check name validity
            nt = namedtuple('_nt', outcols, verbose=False)
            outcols = list(nt._fields)
            if set(outcols) - set(self.names+['nrow__']) != set():
                raise ValueError, "not all outcols are real column names"

        # Get iterators for selected columns
        icols, dtypes = [], []
        for name in outcols:
            if name == "nrow__":
                icols.append(boolarr.wheretrue(limit=limit, skip=skip))
                dtypes.append((name, np.int_))
            else:
                col = self.cols[name]
                icols.append(col.where(boolarr, limit=limit, skip=skip))
                dtypes.append((name, col.dtype))
        dtype = np.dtype(dtypes)
        return self._iter(icols, dtype)

    def __iter__(self):
        return self.iter(0, self.len, 1)

    def iter(self, start=0, stop=None, step=1, outcols=None,
             limit=None, skip=0):
        """
        iter(start=0, stop=None, step=1, outcols=None, limit=None, skip=0)

        Iterator with `start`, `stop` and `step` bounds.

        Parameters
        ----------
        start : int
            The starting item.
        stop : int
            The item after which the iterator stops.
        step : int
            The number of items incremented during each iteration.  Cannot be
            negative.
        outcols : list of strings or string
            The list of column names that you want to get back in results.
            Alternatively, it can be specified as a string such as 'f0 f1' or
            'f0, f1'.  If None, all the columns are returned.  If the special
            name 'nrow__' is present, the number of row will be included in
            output.
        limit : int
            A maximum number of elements to return.  The default is return
            everything.
        skip : int
            An initial number of elements to skip.  The default is 0.

        Returns
        -------
        out : iterable

        See Also
        --------
        where

        """

        # Check outcols
        if outcols is None:
            outcols = self.names
        else:
            if type(outcols) not in (list, tuple, str):
                raise ValueError, "only list/str is supported for outcols"
            # Check name validity
            nt = namedtuple('_nt', outcols, verbose=False)
            outcols = list(nt._fields)
            if set(outcols) - set(self.names+['nrow__']) != set():
                raise ValueError, "not all outcols are real column names"

        # Check limits
        if step <= 0:
            raise NotImplementedError, "step param can only be positive"
        start, stop, step = slice(start, stop, step).indices(self.len)

        # Get iterators for selected columns
        icols, dtypes = [], []
        for name in outcols:
            if name == "nrow__":
                istop = None
                if limit is not None:
                    istop = limit + skip
                icols.append(it.islice(xrange(start, stop, step), skip, istop))
                dtypes.append((name, np.int_))
            else:
                col = self.cols[name]
                icols.append(
                    col.iter(start, stop, step, limit=limit, skip=skip))
                dtypes.append((name, col.dtype))
        dtype = np.dtype(dtypes)
        return self._iter(icols, dtype)

    def _iter(self, icols, dtype):
        """Return a list of `icols` iterators with `dtype` names."""

        icols = tuple(icols)
        namedt = namedtuple('row', dtype.names)
        iterable = it.imap(namedt, *icols)
        return iterable

    def _where(self, boolarr, colnames=None):
        """Return rows where `boolarr` is true as an structured array.

        This is called internally only, so we can assum that `boolarr`
        is a boolean array.
        """

        if colnames is None:
            colnames = self.names
        cols = [self.cols[name][boolarr] for name in colnames]
        dtype = np.dtype([(name, self.cols[name].dtype) for name in colnames])
        result = np.rec.fromarrays(cols, dtype=dtype).view(np.ndarray)

        return result

    def __getitem__(self, key):
        """
        x.__getitem__(key) <==> x[key]

        Returns values based on `key`.  All the functionality of
        ``ndarray.__getitem__()`` is supported (including fancy
        indexing), plus a special support for expressions:

        Parameters
        ----------
        key : string
            The corresponding ctable column name will be returned.  If
            not a column name, it will be interpret as a boolean
            expression (computed via `ctable.eval`) and the rows where
            these values are true will be returned as a NumPy
            structured array.

        See Also
        --------
        ctable.eval

        """

        # First, check for integer
        if isinstance(key, (int, long)):
            # Get a copy of the len-1 array
            ra = self._arr1.copy()
            # Fill it
            ra[0] = tuple([self.cols[name][key] for name in self.names])
            return ra[0]
        # Slices
        elif type(key) == slice:
            (start, stop, step) = key.start, key.stop, key.step
            if step and step <= 0 :
                raise NotImplementedError("step in slice can only be positive")
        # Multidimensional keys
        elif isinstance(key, tuple):
            if len(key) != 1:
                raise IndexError, "multidimensional keys are not supported"
            return self[key[0]]
        # List of integers (case of fancy indexing), or list of column names
        elif type(key) is list:
            if len(key) == 0:
                return np.empty(0, self.dtype)
            strlist = [type(v) for v in key] == [str for v in key]
            # Range of column names
            if strlist:
                cols = [self.cols[name] for name in key]
                return ctable(cols, key)
            # Try to convert to a integer array
            try:
                key = np.array(key, dtype=np.int_)
            except:
                raise IndexError, \
                      "key cannot be converted to an array of indices"
            return np.fromiter((self[i] for i in key),
                               dtype=self.dtype, count=len(key))
        # A boolean array (case of fancy indexing)
        elif hasattr(key, "dtype"):
            if key.dtype.type == np.bool_:
                return self._where(key)
            elif np.issubsctype(key, np.int_):
                # An integer array
                return np.array([self[i] for i in key], dtype=self.dtype)
            else:
                raise IndexError, \
                      "arrays used as indices must be integer (or boolean)"
        # Column name or expression
        elif type(key) is str:
            if key not in self.names:
                # key is not a column name, try to evaluate
                arr = self.eval(key, depth=4)
                if arr.dtype.type != np.bool_:
                    raise IndexError, \
                          "`key` %s does not represent a boolean expression" %\
                          key
                return self._where(arr)
            return self.cols[key]
        # All the rest not implemented
        else:
            raise NotImplementedError, "key not supported: %s" % repr(key)

        # From now on, will only deal with [start:stop:step] slices

        # Get the corrected values for start, stop, step
        (start, stop, step) = slice(start, stop, step).indices(self.len)
        # Build a numpy container
        n = utils.get_len_of_range(start, stop, step)
        ra = np.empty(shape=(n,), dtype=self.dtype)
        # Fill it
        for name in self.names:
            ra[name][:] = self.cols[name][start:stop:step]

        return ra

    def __setitem__(self, key, value):
        """
        x.__setitem__(key, value) <==> x[key] = value

        Sets values based on `key`.  All the functionality of
        ``ndarray.__setitem__()`` is supported (including fancy
        indexing), plus a special support for expressions:

        Parameters
        ----------
        key : string
            The corresponding ctable column name will be set to `value`.  If
            not a column name, it will be interpret as a boolean expression
            (computed via `ctable.eval`) and the rows where these values are
            true will be set to `value`.

        See Also
        --------
        ctable.eval

        """

        # First, convert value into a structured array
        value = utils.to_ndarray(value, self.dtype)
        # Check if key is a condition actually
        if type(key) is bytes:
            # Convert key into a boolean array
            #key = self.eval(key)
            # The method below is faster (specially for large ctables)
            rowval = 0
            for nrow in self.where(key, outcols=["nrow__"]):
                nrow = nrow[0]
                if len(value) == 1:
                    for name in self.names:
                        self.cols[name][nrow] = value[name]
                else:
                    for name in self.names:
                        self.cols[name][nrow] = value[name][rowval]
                    rowval += 1
            return
        # Then, modify the rows
        for name in self.names:
            self.cols[name][key] = value[name]
        return

    def eval(self, expression, **kwargs):
        """
        eval(expression, **kwargs)

        Evaluate the `expression` on columns and return the result.

        Parameters
        ----------
        expression : string
            A string forming an expression, like '2*a+3*b'. The values
            for 'a' and 'b' are variable names to be taken from the
            calling function's frame.  These variables may be column
            names in this table, scalars, carrays or NumPy arrays.
        kwargs : list of parameters or dictionary
            Any parameter supported by the `eval()` first level function.

        Returns
        -------
        out : carray object
            The outcome of the expression.  You can tailor the
            properties of this carray by passing additional arguments
            supported by carray constructor in `kwargs`.

        See Also
        --------
        eval (first level function)

        """

        # Get the desired frame depth
        depth = kwargs.pop('depth', 3)
        # Call top-level eval with cols as user_dict
        return ca.eval(expression, user_dict=self.cols, depth=depth, **kwargs)

    def flush(self):
        """Flush data in internal buffers to disk.

        This call should typically be done after performing modifications
        (__settitem__(), append()) in persistence mode.  If you don't do this,
        you risk loosing part of your modifications.

        """
        for name in self.names:
            self.cols[name].flush()

    def _get_stats(self):
        """
        _get_stats()

        Get some stats (nbytes, cbytes and ratio) about this object.

        Returns
        -------
        out : a (nbytes, cbytes, ratio) tuple
            nbytes is the number of uncompressed bytes in ctable.
            cbytes is the number of compressed bytes.  ratio is the
            compression ratio.

        """

        nbytes, cbytes, ratio = 0, 0, 0.0
        names, cols = self.names, self.cols
        for name in names:
            column = cols[name]
            nbytes += column.nbytes
            cbytes += column.cbytes
        cratio = nbytes / float(cbytes)
        return (nbytes, cbytes, cratio)

    def __str__(self):
        return array2string(self)

    def __repr__(self):
        nbytes, cbytes, cratio = self._get_stats()
        snbytes = utils.human_readable_size(nbytes)
        scbytes = utils.human_readable_size(cbytes)
        header = "ctable(%s, %s)\n" % (self.shape, self.dtype)
        header += "  nbytes: %s; cbytes: %s; ratio: %.2f\n" % (
            snbytes, scbytes, cratio)
        header += "  cparams := %r\n" % self.cparams
        if self.rootdir:
            header += "  rootdir := '%s'\n" % self.rootdir
        fullrepr = header + str(self)
        return fullrepr


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 78
## End:
