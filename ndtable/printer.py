from pprint import pformat
from carray import array2string, set_printoptions as np_setprintoptions

_show_details = True

def set_printoptions(**kwargs):
    global _show_details
    details = kwargs.pop('details', None)

    if details is not None:
        _show_details = details

    np_setprintoptions(**kwargs)

def generic_str(obj, deferred):
    # Do we force str() to render and consequently do a read
    # operation?

    if deferred:
        str_repr = '<Deferred>'
    else:
        str_repr = ''
        for provider in obj.backends:
            str_repr += provider.repr_data()

    return str_repr

def generic_repr(name, obj, deferred):
    """
    Generic pretty printer for NDTable and NDArray.

    Output is of the form::

        Array(3, int32)
          values   := [Numpy(ptr=60597776, dtype=int64, shape=(3,))];
          metadata := Meta(ECLASS=1, );
          layout   := Identity;

    """

    if deferred:
        if _show_details:
            header = "%s\n" % (name)
            header += "  datashape := %s \n" % str(obj._datashape)
            header += "  metadata  := %s \n"  % (pformat(obj._metadata, width=1))
        else:
            header = ''
    else:
        if _show_details:
            header = "%s\n" % (name)
            header += "  datashape := %s \n" % str(obj._datashape)
            header += "  values    := %s \n"  % list(obj.backends)
            header += "  metadata  := %s \n"  % (pformat(obj._metadata, width=1))
            header += "  layout    := %s \n"  % obj._layout.desc
        else:
            header = ''

    # Show the data below
    fullrepr = header + generic_str(obj, deferred)

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
