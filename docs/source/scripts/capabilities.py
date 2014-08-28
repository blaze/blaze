import numpy as np
from blaze import *
from bokeh.plotting import *
from bokeh.objects import HoverTool, ColumnDataSource
from bokeh.embed import *
from bokeh.resources import Resources
from collections import OrderedDict

import itertools

"""
Run this script AFTER you have done make html. It will automatically put the graph where it needs to be. 
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
        'Arithmetic':True,
        'Reductions':True,
        'Selections':True,
        'Grouping':True,
        'Join':True,
        'Sort':True,
        'Distinct':True,
        'Python Mapping':False
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

colormap = [
    "#5EDA9E", "#FFFFFF"
]

ringo = [x for x in capabilities]
paul = [capabilities[x].values() for x in ringo]
john = ['Arithmetic','Reductions', 'Selections', 'Grouping', 'Join', 'Sort', 'Distinct', 'Python Mapping']

def unpacking(i):
    return_list = []
    for x in i:
        for y in x:
            return_list.append(y)
    return return_list

def color_values(a, b):
    colors = []
    for x in paul:
        if x:
            colors.append(colormap[0])
        elif not x:
            colors.append(colormap[1])
    return colors

paul = unpacking(paul)

x, y = zip(*itertools.product(ringo, john))

colors = color_values(colormap, paul)
x = list(x)
y = list(y)

reset_output()

output_file('build/html/capabilities.html', mode='cdn')

source = ColumnDataSource(
    data=dict(
        xname=x,
        yname=y,
        colors=colors,
    )
)

figure()

rect('xname', 'yname', 0.99, 0.99, source=source,
     x_range=ringo, y_range=john,
     x_axis_location="above",
     color='colors', line_color=None,
     tools="resize,hover,previewsave", title="Blaze Capabilities by Backend",
     plot_width=725, plot_height=400)

grid().grid_line_color = None
axis().axis_line_color = None
axis().major_tick_line_color = None
axis().major_label_text_font_size = "5pt"
axis().major_label_standoff = 0

show()      # show the plot'