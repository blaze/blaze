"""
Pipeline that determines phase ordering and execution.
"""

from __future__ import absolute_import, division, print_function
import types


def run_pipeline(func, env, passes):
    """
    Run a sequence of transforms (given as functions or modules) on the
    AIR function.
    """
    for transform in passes:
        func, env = apply_transform(transform, func, env)
    return func, env


def apply_transform(transform, func, env):
    if isinstance(transform, types.ModuleType):
        result = transform.run(func, env)
    else:
        result = transform(func, env)

    _check_transform_result(transform, result)
    return result or (func, env)


def _check_transform_result(transform, result):
    if result is not None and not isinstance(result, tuple):
        if isinstance(transform, types.ModuleType):
            transform = transform.run
        transform = transform.__module__ + '.' + transform.__name__
        raise ValueError(
            "Expected (func, env) result in %r, got %s" % (transform, result))
