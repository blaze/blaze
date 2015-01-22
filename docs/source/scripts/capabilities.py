import numpy as np
from blaze import *
from bokeh.plotting import *
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.embed import *
from bokeh.resources import Resources
from collections import OrderedDict

from toolz import concat
import itertools

"""
Run this script AFTER you have done make html in the same directory you run make html.
It should automatically put the graph where it needs to be.
"""

capabilities = {
    'Streaming Python':{
        'Scalar Expressions':True,
        'Reductions':True,
        'Selections':True,
        'Split-Apply-Combine':True,
        'Join':True,
        'Python Functions':True,
        'SQL Style Indices':False,
        'Column Store': False,
        'Bigger-than-memory data': "Using Streaming",
    },
    'Pandas':{
        'Scalar Expressions':True,
        'Reductions':True,
        'Selections':True,
        'Split-Apply-Combine':True,
        'Join':True,
        'Python Functions':True,
        'SQL Style Indices':False,
        'Column Store': True,
        'Bigger-than-memory data': False,
    },
    'SQL':{
        'Scalar Expressions':"Generally yes, but math is limited in SQLite",
        'Reductions':True,
        'Selections':True,
        'Split-Apply-Combine':True,
        'Join':True,
        'Python Functions':False,
        'SQL Style Indices':True,
        'Column Store': False,
        'Bigger-than-memory data': True,
    },
    'PySpark':{
        'Scalar Expressions':True,
        'Reductions':True,
        'Selections':True,
        'Split-Apply-Combine':True,
        'Join':True,
        'Python Functions':True,
        'SQL Style Indices':False,
        'Column Store': False,
        'Bigger-than-memory data': True,
    },
    'MongoDB':{
        'Scalar Expressions':False,
        'Reductions':True,
        'Selections':"Limited due to lack of scalar expressions",
        'Split-Apply-Combine':"Limited",
        'Join':False,
        'Python Functions':False,
        'SQL Style Indices':True,
        'Column Store': False,
        'Bigger-than-memory data': True,
    },
    'PyTables':{
        'Scalar Expressions': "Using NumExpr",
        'Reductions': True,
        'Selections': "Using NumExpr",
        'Split-Apply-Combine':"Using Streaming Python",
        'Join':False,
        'Python Functions':"Using streaming Python",
        'SQL Style Indices':True,
        'Column Store': False,
        'Bigger-than-memory data': "With fast compressed access",
    },
    'BColz':{
        'Scalar Expressions':"Using chunked NumPy",
        'Reductions':"Using chunked NumPy",
        'Selections':"Using NumExpr",
        'Split-Apply-Combine':"Using streaming Python",
        'Join':False,
        'Python Functions': "Using streaming Python",
        'SQL Style Indices':False,
        'Column Store': True,
        'Bigger-than-memory data': "With fast compressed access",
    }
}

colormap = {True: "#5EDA9E", False: "#FFFFFF"}

backends = ['Streaming Python', 'Pandas', 'PySpark', 'SQL', 'PyTables', 'BColz',
            'MongoDB']
operations = ['Scalar Expressions', 'Selections', 'Reductions',
              'Split-Apply-Combine', 'Join', 'Python Functions', 'SQL Style Indices',
              'Column Store', 'Bigger-than-memory data']

statuses = [capabilities[backend][op] for backend in backends
                                      for op in operations]

x, y = zip(*itertools.product(backends, operations))
x = list(x)
y = list(y)

colors = [colormap[1] if value else colormap[0] for value in statuses]

reset_output()

output_file('build/html/capabilities.html', mode='cdn')


source = ColumnDataSource(
    data=dict(
        xname=x,
        yname=y,
        statuses=statuses,
        colors=colors,
    )
)

figure()

rect('xname', 'yname', 0.99, 0.99, source=source,
     x_range=backends, y_range=list(reversed(operations)),
     x_axis_location="above",
     color='colors', line_color=None,
     tools="hover,previewsave", title="Blaze Capabilities by Backend",
     plot_width=800, plot_height=600)

grid().grid_line_color = None
axis().axis_line_color = None
axis().major_tick_line_color = None
axis().major_label_text_font_size = "5pt"
axis().major_label_standoff = 0

hover = [t for t in curplot().tools if isinstance(t, HoverTool)][0]
hover.tooltips = OrderedDict([
    ('names', '@yname, @xname'),
    ('status', '@statuses'),
])

save()      # show the plot'
