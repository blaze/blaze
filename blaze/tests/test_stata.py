from __future__ import absolute_import, division, print_function

from blaze.stata import resource
import blaze
import os

def test_stata_resource():
    # filename like '/path/to/blaze/examples/data/oil.dta'
    fn = os.path.join(blaze.__path__[0][:-len('blaze')],
                     'examples', 'data', 'oil.dta')
    df = resource(fn)
    assert list(df.columns) == ['year', 'barrels']
