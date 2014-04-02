=========
Tutorials
=========

This chapter goes through a series of tutorials that exercises
different aspects of Blaze on a friendly and step-by-step way.  These
are great if you are learning Blaze, or you want better insight on how
to do specific things with it.

Creating arrays
===============

Building basic arrays
---------------------

Let's start creating a small array:

.. doctest::

  >>> import blaze
  >>> a = blaze.array([2, 3, 4])
  >>> print a
  [2 3 4]

Easy uh?  Blaze tries to follow the NumPy API as much as possible, so
that people will find it easy to start doing basic things.

Now, let's have a look at the representation of the `a` array:

.. doctest::

  >>> a
  array([2, 3, 4],
        dshape='3 * int32')

You are seeing here our first difference with NumPy, and it is the
`dshape` attribute. This is basically the fusion of `shape` and
`dtype` from NumPy.  Such unification of concepts will be
handy for performing advanced operations (like views) in a more
powerful way.

Note that when creating from a Python iterable, a datashape will be
inferred:

.. doctest::

  >>> print(a.dshape)
  3 * int32

In this example, the shape part is '3' while the type part is 'int32'.

Let's create an array of floats:

.. doctest::

  >>> b = blaze.array([1.2, 3.5, 5.1])
  >>> print(b)
  [ 1.2  3.5  5.1]
  >>> print(b.dshape)
  3 * float64

And a bidimensional array:

.. doctest::

  >>> c = blaze.array([ [1, 2], [3, 4] ])
  >>> print(c)
  [[1 2]
   [3 4]]
  >>> print(c.dshape)
  2, 2, int32

or as many dimensions as you like:

.. doctest::

  >>> d = blaze.array([ [ [1, 2], [3, 4] ], [ [5, 6], [7, 8] ] ])
  >>> print(d)
  [[[1 2]
    [3 4]]
  <BLANKLINE>
   [[5 6]
    [7 8]]]
  >>> print(d.dshape)
  2, 2, 2, int32

You can even build a compressed array in-memory:

.. doctest::

  >>> import blz
  >>> ddesc = blaze.BLZ_DDesc(mode='w', bparams=blz.bparams(clevel=5))
  >>> arr = blaze.array([1, 2, 3], ddesc=ddesc)
  >>> print(arr)
  [1 2 3]

It is possible to force the type when creating the array. This
allows a broader selection of types on construction:

.. doctest::

  >>> e = blaze.array([ 1, 2, 3], dshape='3, float32')
  >>> e
  array([ 1.,  2.,  3.],
        dshape='3 * float32')

Note that the dimensions in the datashape when creating from a
collection can be omitted. If that's the case, the dimensions will be
inferred. The following is thus equivalent:

.. doctest::

  >>> f = blaze.array([ 1, 2, 3], dshape='float32')
  >>> f
  array([ 1.,  2.,  3.],
        dshape='3 * float32')

Blaze also supports arrays to be made persistent. This can be achieved
by adding the data descriptor to the `array` constructor:

.. doctest::

  >>> dd = blaze.BLZ_DDesc('myarray.blz', mode='w')
  >>> g = blaze.array([ 1, 2, 3], dshape='float32', ddesc=dd)
  >>> g
  array([ 1.,  2.,  3.],
        dshape='3 * float32')

You can use the persistent array as if it was an in-memory
array. However, it is persistent and it will survive your python
session.

Later on you can gain a reference to the array, even from a
different python session by name, by simply passing the data
descritor to the array constructor:

.. doctest::

  >>> dd = blaze.BLZ_DDesc('myarray.blz', mode='a')
  >>> h = blaze.array(dd)
  >>> h
  array([ 1.,  2.,  3.],
        dshape='3 * float32')

Please note that, as we want to retrieve the existing data, we are
specifying the dataset in 'a'ppend mode, so as not to overwrite it.

A persistent array that has the 'appendable' capability set to True
can be enlarged anytime by using the `blaze.append()` function, e.g.

.. doctest::

  >>> h.ddesc.capabilities.appendable
  True
  >>> blaze.append(h, [4, 5, 6])
  >>> h
  array([ 1.,  2.,  3.,  4.,  5.,  6.],
        dshape='6, float32')

If you are done with the persistent array and want to free
its resources, you can just 'remove' it:

.. doctest::

  >>> h.ddesc.remove()

This will remove the dataset from disk, so it could not be restored in
the future, so if you love your data, be careful with this one.


Evaluation
==========

Performing basic computations
-----------------------------

Performing computations in blaze is a 2 step process. First, you just
use expressions to build a *deferred* array. A *deferred* array,
instead of holding the result, knows how to build that result:

.. doctest::

  >>> a = blaze.array([1, 2, 3])
  >>> a.deferred
  False


.. doctest::

  >>> b = blaze.array([4, 5, 6])
  >>> b.deferred
  False


.. doctest::

  >>> r = a+b
  >>> r.deferred
  True

In order to obtain the results, just call the eval function with the
*deferred* array:

.. doctest::

  >>> result = blaze.eval(r)
  >>> result
  array([5, 7, 9],
        dshape='3 * int32')

So, why this extra step? why the need to evaluate instead of just
generating the result directly from a+b? The answer is a bit
complex. Making a long story short, using the *deferred* array allows
building a complex expression and optimize it as a whole before
execution. This allows removing the need of arrays for intermediate
results, as well as the need to perform several passes on data. A
short answer is that it allows blaze to perform better with big data sets.

Also, having an explicit evaluation method gives us a chance to
specify a few parameters telling how the resulting array should be
built. As can be seen in the array creation tutorial, an array can be
made in-memory, compressed in-memory or it can even be backed on the
file-system. We can eval directly to a persistent array:

.. doctest::

  >>> dd = blaze.BLZ_DDesc('res.blz', mode='w')
  >>> result = blaze.eval(r, ddesc=dd)

In this sample we have used two small in-memory arrays to illustrate
execution. The same code can work for large arrays that are 'opened'
instead of being created/read, allowing the easy evaluation of
expression that is effectively out-of-core::

  >>> ba1 = blaze.array(blaze.BLZ_DDesc('big_array1.blz'))
  >>> ba2 = blaze.array(blaze.BLZ_DDesc('big_array2.blz'))
  >>> dd_res = blaze.BLZ_DDesc('big_result.blz', mode='w')
  >>> res = blaze.eval(ba1+ba2, ddesc=dd_res)

So it is possible to build complex array expressions that can be
executed without building huge intermediate arrays. It is also
possible to use persistent arrays or in-memory arrays as your operands
(or a mix of both, as they are all Blaze arrays). You are also able to
specify what kind of array you want for your result.
