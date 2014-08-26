from __future__ import absolute_import, division, print_function
from dynd import nd
import pandas as pd
import os
from ..dispatch import dispatch
from ..data import DataDescriptor, CSV


def resource(a, **kwargs):
    if os.path.isfile(a) and a.endswith("csv"):
        return CSV(a, **kwargs)
    else:
        raise NotImplementedError(
            "Blaze can't read '%s' yet!" % (a))
