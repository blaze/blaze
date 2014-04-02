'''Sample module showing the creation of blaze arrays'''

from __future__ import absolute_import, division, print_function

import blaze

def print_section(a_string, level=0):
    spacing = 2 if level == 0 else 1
    underline = ['=', '-', '~', ' '][min(level,3)]

    print ('%s%s\n%s' % ('\n'*spacing,
                         a_string,
                         underline*len(a_string)))


print_section('building basic arrays')
# It is possible to build arrays from python lists
a = blaze.array([ 2, 3, 4 ])

# Arrays can be printed
print(a)

# The array will have a datashape. A datashape is a combination of the
# shape and dtype concept found in numpy. Note that when creating from
# a Python iterable, a datashape will be inferred.
print(a.dshape)

b = blaze.array([1.2, 3.5, 5.1])
print(b)
print(b.dshape)

# Arrays can be bi-dimensional
print_section('going 2d', level=1)
c = blaze.array([ [1, 2], [3, 4] ])
print(c)
print(c.dshape)

# or as many dimensions as you like
print_section('going 3d', level=1)
d = blaze.array([ [ [1, 2], [3, 4] ], [ [5, 6], [7, 8] ] ])
print(d)
print(d.dshape)

# --------------------------------------------------------------------

print_section ('building compressed in-memory arrays')

# A compressed array (backed by BLZ):
import blz
datadesc = blaze.BLZ_DDesc(mode='w', bparams=blz.bparams(clevel=5))
arr = blaze.array([1,2,3])
print(arr)

# --------------------------------------------------------------------

print_section('Explicit types in construction')
# It is possible to force a type in a given array. This allows a
# broader selection of types on construction.
e =  blaze.array([1, 2, 3], dshape='3 * float32')
print(e)

# Note that the dimensions in the datashape when creating from a
# collection can be omitted. If that's the case, the dimensions will
# be inferred. The following is thus equivalent:

f = blaze.array([1, 2, 3], dshape='float32')
print(f)

# --------------------------------------------------------------------

print_section('Alternative  constructors')

# Arrays can be created to be all zeros:
g = blaze.zeros('10 * 10 * int32')
print(g)

# All ones
h = blaze.ones('10 * 10 * float64')
print(h)

# --------------------------------------------------------------------

print_section('Indexing')

print_section('Indexing for read', level=1)
print ('starting with a 4d array')
array4d = blaze.ones('10 * 10 * 10 * 10 * float32')

def describe_array(label, array):
    print(label)
    print('dshape: ', array.dshape)
    print(array)

describe_array('base', array4d)
describe_array('index once', array4d[3])
describe_array('index twice', array4d[3,2])
describe_array('index thrice', array4d[3,2,4])
describe_array('index four times', array4d[3,2,4,1])


print_section('Indexing for write', level=1)
array4d[3,2,4,1] = 16.0

describe_array('base', array4d)
describe_array('index once', array4d[3])
describe_array('index twice', array4d[3,2])
describe_array('index thrice', array4d[3,2,4])
describe_array('index four times', array4d[3,2,4,1])

array4d[3,2,1] = 3.0

describe_array('base', array4d)
describe_array('index once', array4d[3])
describe_array('index twice', array4d[3,2])
describe_array('index thrice', array4d[3,2,4])
describe_array('index four times', array4d[3,2,4,1])


del describe_array

# --------------------------------------------------------------------

print_section('Persisted arrays')

# Create an empty array on-disk
dname = 'persisted.blz'
datadesc = blaze.BLZ_DDesc(dname, mode='w')
p = blaze.zeros('0 * float64', ddesc=datadesc)
# Feed it with some data
blaze.append(p, range(10))

print(repr(datadesc))
print('Before re-opening:', p)

# Re-open the dataset in 'r'ead-only mode
datadesc = blaze.BLZ_DDesc(dname, mode='r')
p2 = blaze.array(datadesc)

print('After re-opening:', p2)

# Remove the dataset on-disk completely
datadesc.remove()
