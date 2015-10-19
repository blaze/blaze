Time Series Operations
======================

Blaze is starting to grow some time series related expressions.
These are generally inspired by pandas API, but are different in cases
where the pandas API is too big or doesn't work well with the way
Blaze works.

A notable difference is that blaze's time series operations are *not*
centered around the notion of an index.

.. note::

   Time series functionality is very new. Please keep this in mind
   when writing production code that uses these new features.


So far, we have :func:`~blaze.expr.collections.shift` since PR :issue:`1266`.

On the short-term roadmap are rolling aggregates and resample for both the
Pandas and SQL backends.
