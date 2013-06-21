=========
Tutorials
=========

This chapter goes through a series of tutorials that exercises different aspects of Blaze on a friendly and step-by-step way.  These are great if you are learning Blaze, or you want better insight on how to do specific things with it.

Creating arrays
===============

Building basic arrays
---------------------

Let's start creating a small array:

  >>> import blaze
  >>> a = blaze.array([2, 3, 4])
  >>> print a
  [2 3 4]

Easy uh?  Blaze tries to follow the NumPy API as much as possible, so
that people will find it easy to start doing basic things.

Now, let's have a look at the representation of the `a` array::

  >>> a
  array([2, 3, 4],
        dshape='3, int64')

You are seeing here at a first difference with NumPy, and it is the
`dshape` attribute, which is basically the fusion of `shape` and
`dtype` concepts for NumPy.  Such unification of concepts will be
handy for performing advanced operations (like views) in a more
powerful way.

Note that when creating from a Python iterable, a datashape will be
inferred::

  >>> print(a.dshape)
  3, int64

In this example, the shape part is '3' while the type part is 'int64'.

Let's create an array of floats::

  >>> b = blaze.array([1.2, 3.5, 5.1])
  >>> print(b)
  [ 1.2  3.5  5.1]
  >>> print(b.dshape)
  3, float64

And a bidimensional array::

  >>> c = blaze.array([ [1, 2], [3, 4] ]) 
  >>> print(c)
  [[1 2]
   [3 4]]
  >>> print(c.dshape)
  2, 2, int64

or as many dimensions as you like::

  >>> d = blaze.array([ [ [1, 2], [3, 4] ], [ [5, 6], [7, 8] ] ])
  >>> print(d)
  [[[1 2]
    [3 4]]

   [[5 6]
    [7 8]]]
  >>> print(d.dshape)
  2, 2, 2, int64

You can even build a compressed array in-memory::

  >>> blz = blaze.array([1,2,3], caps={'compress': True})
  >>> print(blz)
  [1 2 3]

It is possible to force a type in a given array. This allows a broader
selection of types on construction::

  >>> e = blaze.array([ 1, 2, 3], dshape='3, float32') 
  >>> e
  array([ 1.,  2.,  3.],
        dshape='3, float32')

Note that the dimensions in the datashape when creating from a
collection can be omitted. If that's the case, the dimensions will be
inferred. The following is thus equivalent::

  >>> f = blaze.array([ 1, 2, 3], dshape='float32')
  >>> f
  array([ 1.,  2.,  3.],
        dshape='3, float32')

Blaze also supports arrays to be made persistent. This can be achieved
by adding the persist keyword parameter to an array constructor::

  >>> g = blaze.array([ 1, 2, 3], dshape='float32', persist=blaze.Persist('myarray'))
  >>> g
  array([ 1.,  2.,  3.],
        dshape='3, float32')

You can use the persistent array as if it was an in-memory
array. However, it is persistent and it will survive your python
session. Later you can gain a reference to the array, even from a
different python session by name, using the open function::

  >>> f = blaze.open(blaze.Persist('myarray'))
  >>> f
  array([ 1.,  2.,  3.],
        dshape='3, float32')

A persistent array is stored on non-volatile storage (currently the
filesystem). That means that there are system resources allocated to
store that array, even when you exit your python session. If you are
done with the persistent array and want to keep its resources, you can
just 'drop' it::

  >>> f = blaze.drop(blaze.Persist('myarray'))

Note that after dropping a persistent array this way, any 'open'
version you may had of it will no longer be valid. You won't be able
to reopen it either. It is effectively deleted.

