import os
import iopro

from blaze import Table, mean, std, params, select, open
from blaze.algo.select import select2

adapter = iopro.text_adapter(
    'noaa_gsod_example.op',
    header=1,
    infer_types=False,
    field_names=['x', 'y', 'z', 'w']
)
adapter.set_field_types({0:'u8', 1:'u8', 2:'u8', 3:'f8'})


def test_simple():
    if not os.path.exists('./noaa_data'):
        p = params(clevel=5, storage='./noaa_data')

        t = Table([], dshape='{f0: int, f1:int, f2:int, f3:float}', params=p)

        # TODO: chunkwise copy
        t.append(adapter[:])
        t.commit()
    else:
        t = open('ctable://noaa_data')

    print '--------------------------------------'
    print 'mean', mean(t, 'f3')
    print 'std', std(t, 'f2')
    print '--------------------------------------'

    qs1 = select(t, lambda x: x > 80000, 'f0')
    qs2 = select2(t, lambda x,y: x > y, ['f0', 'f1'])

    result = t[qs1]

if __name__ == '__main__':
    test_simple()
