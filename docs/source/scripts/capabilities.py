from bokeh.plotting import reset_output, figure, save, output_file
from bokeh.models import HoverTool, ColumnDataSource
from collections import OrderedDict

import itertools

"""
Run this script AFTER you have done make html in the same directory you run
make html. It should automatically put the graph where it needs to be.
"""

capabilities = {
    'Streaming Python': {
        'Scalar Expressions': True,
        'Reductions': True,
        'Selections': True,
        'Split-Apply-Combine': True,
        'Join': True,
        'Python Functions': True,
        'SQL Style Indices': False,
        'Column Store': False,
        'Bigger-than-memory data': "Using Streaming",
    },
    'Pandas': {
        'Scalar Expressions': True,
        'Reductions': True,
        'Selections': True,
        'Split-Apply-Combine': True,
        'Join': True,
        'Python Functions': True,
        'SQL Style Indices': False,
        'Column Store': True,
        'Bigger-than-memory data': False,
    },
    'SQL': {
        'Scalar Expressions': "Generally yes, but math is limited in SQLite",
        'Reductions': True,
        'Selections': True,
        'Split-Apply-Combine': True,
        'Join': True,
        'Python Functions': False,
        'SQL Style Indices': True,
        'Column Store': False,
        'Bigger-than-memory data': True,
    },
    'PySpark': {
        'Scalar Expressions': True,
        'Reductions': True,
        'Selections': True,
        'Split-Apply-Combine': True,
        'Join': True,
        'Python Functions': True,
        'SQL Style Indices': False,
        'Column Store': False,
        'Bigger-than-memory data': True,
    },
    'MongoDB': {
        'Scalar Expressions': False,
        'Reductions': True,
        'Selections': "Limited due to lack of scalar expressions",
        'Split-Apply-Combine': "Limited",
        'Join': False,
        'Python Functions': False,
        'SQL Style Indices': True,
        'Column Store': False,
        'Bigger-than-memory data': True,
    },
    'PyTables': {
        'Scalar Expressions': "Using NumExpr",
        'Reductions': True,
        'Selections': "Using NumExpr",
        'Split-Apply-Combine': "Using Streaming Python",
        'Join': False,
        'Python Functions': "Using streaming Python",
        'SQL Style Indices': True,
        'Column Store': False,
        'Bigger-than-memory data': "With fast compressed access",
    },
    'BColz': {
        'Scalar Expressions': "Using chunked NumPy",
        'Reductions': "Using chunked NumPy",
        'Selections': "Using NumExpr",
        'Split-Apply-Combine': "Using streaming Python",
        'Join': False,
        'Python Functions': "Using streaming Python",
        'SQL Style Indices': False,
        'Column Store': True,
        'Bigger-than-memory data': "With fast compressed access",
    }
}

if __name__ == '__main__':
    colormap = {True: "#5EDA9E", False: "#FFFFFF"}

    backends = sorted(capabilities.keys())
    operations = sorted(tuple(capabilities.values())[0].keys())

    statuses = [capabilities[backend][op] for backend in backends
                for op in operations]

    x, y = map(list, zip(*itertools.product(backends, operations)))

    colors = [colormap[1] if value else colormap[0] for value in statuses]

    reset_output()

    output_file('source/_static/html/capabilities.html')


    source = ColumnDataSource(
        data=dict(
            xname=x,
            yname=y,
            statuses=statuses,
            colors=colors,
        )
    )

    p = figure(title="Blaze Capabilities by Backend",
               tools='hover,save',
               x_range=backends,
               y_range=list(reversed(operations)),
               x_axis_location="above")
    p.plot_width = 800
    p.plot_height = 600

    p.rect('xname', 'yname', 0.99, 0.99, source=source,
           color='colors')

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0

    hover = p.select(dict(type=HoverTool))[0]
    hover.tooltips = OrderedDict([
        ('names', '@yname, @xname'),
        ('status', '@statuses'),
    ])

    save(p)
