from ..datashape import DataShape, Record, Fixed, Var, CType, String, JSON
#from blaze_server_config import jinja_env
#from jinja2 import Template

def json_comment(array_url):
    return '<font style="font-size:x-small"> # <a href="' + \
        array_url + '?r=data.json">JSON</a></font>\n'

def render_datashape_recursive(base_url, ds, indent):
    result = ''

    if isinstance(ds, DataShape):
        for dim in ds[:-1]:
            if isinstance(dim, Fixed):
                result += ('%d, ' % dim)
            elif isinstance(dim, Var):
                result += 'var, '
            else:
                raise TypeError('Cannot render datashape with dimension %r' % dim)
        result += render_datashape_recursive(base_url, ds[-1], indent)
    elif isinstance(ds, Record):
        result += '{' + json_comment(base_url)
        for fname, ftype in zip(ds.names, ds.types):
            child_url = base_url + '.' + fname
            child_result = render_datashape_recursive(child_url,
                            ftype, indent + '  ')
            result += (indent + '  ' +
                '<a href="' + child_url + '">' + str(fname) + '</a>'
                ': ' + child_result + ';')
            if isinstance(ftype, Record):
                result += '\n'
            else:
                result += json_comment(child_url)
        result += (indent + '}')
    elif isinstance(ds, (CType, String, JSON)):
        result += str(ds)
    else:
        raise TypeError('Cannot render datashape %r' % ds)
    return result

def render_datashape(base_url, ds):
    print('base url is %s' % base_url)
    result = render_datashape_recursive(base_url, ds, '')
    result = '<pre>\ntype <a href="' + base_url + \
            '?r=datashape">BlazeDataShape</a> = ' + result + '\n</pre>'
    return result
