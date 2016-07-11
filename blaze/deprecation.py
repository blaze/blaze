from __future__ import print_function, division, absolute_import
import warnings
from functools import wraps


def deprecated(version, replacement=None):
    """Define a deprecation decorator.
    An optional `replacement` should refer to the new API to be used instead.

    Example:
      @deprecated('1.1')
      def old_func(): ...

      @deprecated('1.1', 'new_func')
      def old_func(): ..."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            msg = "\"{}\" has been deprecated in version {} and will be removed in a future version."
            if replacement:
                msg += "\n Use \"{}\" instead."
            warnings.warn(msg.format(func.__name__, version, replacement),
                          category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wraps(func)(wrapper)

    return decorator
