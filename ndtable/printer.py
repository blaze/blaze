# TODO: Talk with Francesc he has a robust array printer in
# carray we can use.

from pprint import pformat

def generic_repr(name, obj, deferred):
    """
    Generic pretty printer for NDTable and NDArray.

    Output is of the form::

        Array(3, int32)
          values   := [Numpy(ptr=60597776, dtype=int64, shape=(3,))];
          metadata := Meta(ECLASS=1, );
          layout   := Identity;

    """
    header = "%s\n" % (name)
    header += "  datashape := %s \n" % str(obj._datashape)
    header += "  values    := %s \n"  % list(obj.backends)
    header += "  metadata  := %s \n"  % (pformat(obj._metadata, width=1))
    header += "  layout    := %s \n"  % obj._layout.desc

    # Do we force str() to render and consequently do a read
    # operation?
    if deferred:
        fullrepr = header + '<Deferred>'
    else:
        fullrepr = header + str(obj)

    return fullrepr

#------------------------------------------------------------------------
# Console
#------------------------------------------------------------------------

def array2string(a):
    return ''

def table2string(t):
    return ''

#------------------------------------------------------------------------
# IPython Notebook
#------------------------------------------------------------------------

def array2html(a):
    html = "<table></table>"

    return ('<div style="max-height:1000px;'
            'max-width:1500px;overflow:auto;">\n' +
            html + '\n</div>')

def table2html(t):
    html = "<table></table>"

    return ('<div style="max-height:1000px;'
            'max-width:1500px;overflow:auto;">\n' +
            html + '\n</div>')
