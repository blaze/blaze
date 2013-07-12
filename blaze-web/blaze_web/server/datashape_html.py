from dynd import nd, ndt
#from blaze_server_config import jinja_env
#from jinja2 import Template

def json_comment(array_url):
    return '<font style="font-size:x-small"> # <a href="' + \
        array_url + '?r=data.json">JSON</a></font>\n'

def render_dynd_datashape_recursive(base_url, arr, indent):
    result = ''
    if isinstance(arr, ndt.type):
        dt = arr.value_type
    else:
        dt = nd.type_of(arr).value_type

    if dt.kind == 'struct':
        result += '{' + json_comment(base_url)
        for farr, fname in zip(dt.field_types, dt.field_names):
            farr = nd.as_py(farr)
            child_url = base_url + '.' + str(fname)
            child_result = render_dynd_datashape_recursive(child_url,
                            farr, indent + '  ')
            result += (indent + '  ' +
                '<a href="' + child_url + '">' + str(fname) + '</a>'
                ': ' + child_result + ';')
            dt = nd.dtype_of(farr) if isinstance(farr, nd.array) else farr
            if dt.kind != 'struct':
                result += json_comment(child_url)
            else:
                result += '\n'
        result += (indent + '}')
    elif dt.kind == 'uniform_dim':
        if dt.type_id in ['strided_dim', 'var_dim']:
            if isinstance(arr, ndt.type):
                result += 'var, '
            else:
                result += ('%d, ' % len(arr))
        elif dt.type_id == 'fixed_dim':
            result += ('%d, ' % dt.fixed_dim_size)
        else:
            raise TypeError('Unrecognized DyND uniform array type ' + str(dt))
        # Descend to the element type
        if isinstance(arr, ndt.type):
            arr = arr.element_type
        elif len(arr) == 1:
            # If there's only one element in the array, can
            # keep using the array sizes in the datashape
            arr = arr[0]
        else:
            arr = nd.type_of(arr).element_type
        result += render_dynd_datashape_recursive(base_url, arr, indent)
    elif dt.kind in ['bool', 'int', 'uint', 'real', 'datetime', 'json', 'string']:
        result += str(dt.dshape)
    else:
        raise TypeError('Unrecognized DyND type ' + str(dt))
    return result

def render_dynd_datashape(base_url, arr):
    result = render_dynd_datashape_recursive(base_url, arr, '')
    result = '<pre>\ntype <a href="' + base_url + \
            '?r=datashape">BlazeDataShape</a> = ' + result + '\n</pre>'
    return result
