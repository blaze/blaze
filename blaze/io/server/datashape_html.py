from datashape import DataShape, Record, Fixed, Var, CType, String, JSON
from jinja2 import Template


json_comment_templ = Template("""<font style="font-size:x-small"> # <a href="{{base_url}}?r=data.json">JSON</a></font>

""")

datashape_outer_templ = Template("""
<pre>
type <a href="{{base_url}}?r=datashape">BlazeDataShape</a> = {{ds_html}}
</pre>
""")


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
        result += '{' + json_comment_templ.render(base_url=base_url)
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
                result += json_comment_templ.render(base_url=child_url)
        result += (indent + '}')
    elif isinstance(ds, (CType, String, JSON)):
        result += str(ds)
    else:
        raise TypeError('Cannot render datashape %r' % ds)
    return result


def render_datashape(base_url, ds):
    ds_html = render_datashape_recursive(base_url, ds, '')
    return datashape_outer_templ.render(base_url=base_url, ds_html=ds_html)
