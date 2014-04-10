# Rolling Reduction UFuncs



## References

* https://en.wikipedia.org/wiki/Moving_average
* http://pandas.pydata.org/pandas-docs/stable/computation.html#moving-rolling-statistics-moments

## Pandas Implementation

The core of Pandas' rolling sum and rolling mean are
here in the codebase:

https://github.com/pydata/pandas/blob/master/pandas/algos.pyx#L840

Characteristics of the implementation include:

* Outputs NaNs until the window size is reached.
* Keeps track of the number of non-NaNs encountered
* Has parameter `minp` controlling the number of
  non-NaN values required to return a finite value.
* 
* The data is preprocessed with a 'kill_inf'
  option converting infinities to NaN.
  * This has implications, such as rolling_max
    incorrectly giving NaN instead of +inf
    where +inf appears, or NaN instead of the
    valid finite values where -inf appears.

## Miscellaneous Notes

* In science and engineering, a centred window is
  common, but in financial applications a trailing
  window is used. I assume the latter is because in
  you don't want the statistic to include information
  from the future in a time series analysis.

See also

* [Elementwise Reduction UFuncs](elwise-reduction-ufuncs.md)
