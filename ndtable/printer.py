from pprint import pformat
from carray import array2string, set_printoptions as np_setprintoptions

_show_details = True

def set_printoptions(**kwargs):
    global _show_details
    details = kwargs.pop('details', None)

    if details is not None:
        _show_details = details

    np_setprintoptions(**kwargs)

def generic_repr(name, obj, deferred):
    """
    Generic pretty printer for NDTable and NDArray.

    Output is of the form::

        Array(3, int32)
          values   := [Numpy(ptr=60597776, dtype=int64, shape=(3,))];
          metadata := Meta(ECLASS=1, );
          layout   := Identity;

    """
    if _show_details:
        header = "%s\n" % (name)
        header += "  datashape := %s \n" % str(obj._datashape)
        header += "  values    := %s \n"  % list(obj.backends)
        header += "  metadata  := %s \n"  % (pformat(obj._metadata, width=1))
        header += "  layout    := %s \n"  % obj._layout.desc
    else:
        header = ''

    # Do we force str() to render and consequently do a read
    # operation?
    if deferred:
        fullrepr = header + '<Deferred>'
    else:
        fullrepr = header
        for provider in obj.backends:
            fullrepr += provider.repr_data()

    return fullrepr

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
