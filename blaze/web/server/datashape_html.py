from dynd import nd, ndt
#from blaze_server_config import jinja_env
from jinja2 import Template

def json_comment(array_url):
    return '<font style="font-size:x-small"> # <a href="' + \
        array_url + '?r=data.json">JSON</a></font>\n'

def render_dynd_datashape_recursive(base_url, arr, indent):
    result = ''
    if type(arr) is nd.dtype:
        dt = arr
    else:
        dt = arr.dtype

    if dt.kind == 'struct':
        result += '{' + json_comment(base_url)
        field_names = dt.field_names.as_py()
        for i, fname in enumerate(field_names):
            farr = arr[i]
            if type(farr) is not nd.dtype:
                farr = farr.dtype
            child_url = base_url + '.' + str(fname)
            child_result = render_dynd_datashape_recursive(child_url,
                            farr, indent + '  ')
            result += (indent + '  ' +
                '<a href="' + child_url + '">' + str(fname) + '</a>'
                ': ' + child_result + ';')
            if farr.kind != 'struct':
                result += json_comment(child_url)
        result += (indent + '}')
    elif dt.kind == 'uniform_array':
        if dt.type_id == 'strided_array':
            if (type(arr) is not nd.dtype):
                result += (str(arr.shape[0]) + ', ')
            else:
                result += 'VarDim, '
        elif dt.type_id == 'fixedarray':
            result += (str(dt.fixed_dim_size) + ', ')
        elif dt.type_id == 'var_array':
            result += 'VarDim, '
        else:
            raise TypeError('Unrecognized DyND uniform array type ' + str(dt))
        if (type(arr) is not nd.dtype):
            arr = arr.dtype
        result += render_dynd_datashape_recursive(base_url, arr[0], indent)
    elif dt.kind in ['bool', 'int', 'uint', 'real', 'datetime', 'json', 'string']:
        result += str(dt)
    else:
        raise TypeError('Unrecognized DyND type ' + str(dt))
    return result

def render_dynd_datashape(base_url, arr):
    result = render_dynd_datashape_recursive(base_url, arr, '')
    result = '<pre>\ntype <a href="' + base_url + \
            '?r=datashape">BlazeDataShape</a> = ' + result + '\n</pre>'
    return result
