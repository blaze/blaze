import pytest
import sqlalchemy as sa

from blaze import CSV, resource
from blaze.utils import chmod, filetext, WORLD_READABLE


@pytest.yield_fixture
def csv():
    with filetext('1,2\n10,20\n100,200', '.csv') as f:
        with chmod(f, flags=WORLD_READABLE) as g:
            yield CSV(g, columns=list('ab'))


@pytest.yield_fixture
def sql(csv, request):
    name = 'test_table'
    s = resource(request.module.url, name, schema=csv.schema)
    engine = s.engine
    yield s
    metadata = sa.MetaData()
    metadata.reflect(engine, only=[s.tablename])
    t = metadata.tables[s.tablename]
    t.drop(engine)
