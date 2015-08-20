=====================
What Blaze Doesn't Do
=====================

Blaze occasionally suffers from over-hype.  The terms *Big-Data* and *Pandas*
inevitably conflate in people's minds to become something unattainable and lead
to disappointment.  Blaze is limited;  learning those limitations can direct
you to greater productivity.

First and foremost, Blaze does not replace Pandas.  Pandas will always be more
feature rich and more mature than Blaze.  There are things that you simply
can't do if you want to generalize out of memory.

If your data fits nicely in memory then use NumPy/Pandas.  Your data probably
fits nicely in memory.


Some concrete things Blaze doesn't do
-------------------------------------

1.  Clean unstructured data.  Blaze only handles analytic queries on structured
    data.
2.  Most things in SciPy.  Including things like FFT, and gradient descent.
3.  Most things in SciKit Learn/Image/etc..
4.  Statistical inference - We invite you to build this (this one is actually pretty doable.)
5.  Parallelize your existing Python code
6.  Replace Spark - Blaze may operate on top of Spark, it doesn't compete with it.
7.  Compute quickly - Blaze uses other things to compute, it doesn't compute
    anything itself.  So asking questions about how fast Blaze is are
    determined entirely by how fast other things are.


That's not to say that these can't be done
------------------------------------------

Blaze aims to be a foundational data interface like ``numpy/pandas``
rather than try to implement the entire PyData stack (``scipy, scikit-learn``,
etc..)  Only by keeping scope small do we have a chance at relevance.

Of course, others can build off of Blaze in the same way that ``scipy`` and
``scikit-learn`` built off of ``numpy/pandas``.  Blaze devs often also do this
work (it's important) but we generally don't include it in the Blaze library.

It's also worth mentioning that different classes of algorithms work well on
small vs large datasets.  It could be that the algorithm that you like most may
not easily extend beyond the scope of memory.  A direct translation of
scikit-learn algorithms to Blaze would likely be computationally disastrous.


What Blaze Does
---------------

Blaze is a query system that looks like NumPy/Pandas.  You write Blaze
queries, Blaze translates those queries to something else (like SQL), and ships
those queries to various database to run on other people's fast code.  It
smoothes out this process to make interacting with foreign data as accessible
as using Pandas.  This is actually quite difficult.

Blaze increases human accessibility, not computational performance.


But we work on other things
---------------------------

Blaze devs interact with a lot of other computational systems.  Sometimes we
find holes where systems should exist, but don't.  In these cases we may write
our own computational system.  In these cases we naturally hook up Blaze to
serve as a front-end query system.  We often write about these experiments.

As a result you may see us doing some of the things we just said "Blaze doesn't
do".  These things aren't Blaze (but you *can* use Blaze to do them easily.)
