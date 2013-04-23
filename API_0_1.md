API 0.1
=======

```python
def open(uri, mode='a'):
    """Open a Blaze object via an `uri` (Uniform Resource Identifier).

    Parameters
    ----------
    uri : str
        Specifies the URI for the Blaze object.  It can be a regular file too.
        The URL scheme indicates the storage type:

          * barray: BLZ array
          * btable: BLZ table
          * sqlite: SQLite table (the URI 'sqlite://' creates in-memory table)

        If no URI scheme is given, carray is assumed.

    mode : the open mode (string)
        Specifies the mode in which the object is opened.  The supported
        values are:

          * 'r' for read-only
          * 'w' for emptying the previous underlying data
          * 'a' for allowing read/write on top of existing data

    Returns
    -------
    out : an Array or Table object.

    """
```


```python
def zeros(dshape, params=None):
    """ Create an Array and fill it with zeros.

    Parameters
    ----------
    dshape : str, blaze.dshape instance
        Specifies the datashape of the outcome object.
    params : blaze.params object
        Any parameter supported by the backend library.

    Returns
    -------
    out : an Array object.

    """
```

```python
def ones(dshape, params=None):
    """ Create an Array and fill it with ones.

    Parameters
    ----------
    dshape : str, blaze.dshape instance
        Specifies the datashape of the outcome object.
    params : blaze.params object
        Any parameter supported by the backend library.

    Returns
    -------
    out : an Array object.

    """
```
