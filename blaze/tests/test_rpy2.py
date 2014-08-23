from __future__ import absolute_import

from blaze.rpy2 import into
import pandas as pd
from rpy2.robjects import Sexp
from rpy2.robjects.vectors import DataFrame

def test_rpy2_into():
    pdf = pd.DataFrame({'a': [1,2,3], 'b': [4,5,5.5]})
    rdf = into(Sexp, pdf)
    assert isinstance(rdf, DataFrame)

    pdf2 = into(pd.DataFrame, rdf)
    # This could still be better prolly -DJC
    assert (pdf2 == pdf).all().all()

def test_into_list():
    pdf = pd.DataFrame({'a': [1,2,3], 'b': [4,5,5.5]})
    rdf = into(Sexp, pdf)
    L = into(list, rdf)

    assert isinstance(L, list)
    assert L == [(1, 4), (2, 5), (3, 5.5)]
