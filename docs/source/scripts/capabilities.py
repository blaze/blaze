import numpy as np
from blaze import *
from bokeh.plotting import *
from bokeh.objects import HoverTool, ColumnDataSource
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
        'Arithmetic':True,
        'Reductions':True,
        'Selections':True,
        'Grouping':True,
        'Join':True,
        'Sort':True,
        'Distinct':True,
        'Python Mapping':True
    },
    'Pandas':{
        'Arithmetic':True,
        'Reductions':True,
        'Selections':True,
        'Grouping':True,
        'Join':True,
        'Sort':True,
        'Distinct':True,
        'Python Mapping':True
    },
    'SQL':{
        'Arithmetic':"Generally yes, but math is limited in SQLite",
        'Reductions':True,
        'Selections':True,
        'Grouping':True,
        'Join':True,
        'Sort':True,
        'Distinct':True,
        'Python Mapping':"Only in Postgres"
    },
    'Spark':{
        'Arithmetic':False,
        'Reductions':True,
        'Selections':True,
        'Grouping':True,
        'Join':True,
        'Sort':True,
        'Distinct':True,
        'Python Mapping':False
    },
    'Mongo':{
        'Arithmetic':False,
        'Reductions':True,
        'Selections':True,
        'Grouping':True,
        'Join':False,
        'Sort':True,
        'Distinct':True,
        'Python Mapping':False
    },
    'PyTables':{
        'Arithmetic':True,
        'Reductions':True,
        'Selections':True,
        'Grouping':False,
        'Join':False,
        'Sort':True,
        'Distinct':False,
        'Python Mapping':False
    },
    'BColz':{
        'Arithmetic':False,
        'Reductions':True,
        'Selections':True,
        'Grouping':True,
        'Join':False,
        'Sort':False,
        'Distinct':True,
        'Python Mapping':False
    }
}

colormap = {True: "#5EDA9E", False: "#FFFFFF"}

xnames = [x for x in capabilities]
statuses = [capabilities[x].values() for x in xnames]
ynames = [capabilities['SQL'].keys()]

statuses = list(concat(statuses))
ynames = list(concat(ynames))
x, y = zip(*itertools.product(xnames, ynames))

colors = [colormap[0] if not value else colormap[1] for value in statuses]
x = list(x)
y = list(y)

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
     x_range=xnames, y_range=list(reversed(ynames)),
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
