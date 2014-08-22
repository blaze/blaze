from __future__ import absolute_import, division, print_function
from dynd import nd
import pandas as pd
import os
from ..dispatch import dispatch



@dispatch(object)
def resource(a, **kwargs):
    """
    Dispatch a path-like object to the approprioate resource

    """
    raise NotImplementedError(
        "Blaze does not know a resource for the following type"
        "\n%s " % (type(a).__name__))


@dispatch(str)
def resource(a, **kwargs):
    if os.path.isfile(a) and a.endswith("csv"):
        return pd.read_csv(a, **kwargs)
    else:
        raise NotImplementedError(
            "Blaze can't read '%s' yet!" % (a))
