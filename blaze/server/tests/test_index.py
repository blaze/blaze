from blaze.server.index import *

def test_js_to_tuple():
    data = [(1, 1),
            ('name', 'name'),
            ([1, 2], (1, 2)),
            ({'start': 0, 'stop': 10}, slice(0, 10)),
            ({'start': 0, 'stop': None, 'step': 2}, slice(0, None, 2)),
            ([1, {'start': 0, 'stop': 10}], (1, slice(0, 10))),
            ([{'start': 0, 'stop': 10}, ['name', 'amount']],
                    (slice(0, 10), ['name', 'amount']))]

    for a, b in data:
        if parse_index(a) != b:
            print(a)
            print(b)
            assert False

    for a, b in data:
        if emit_index(b) != a:
            print(a)
            print(b)
            assert False

    assert parse_index([{'start': 0, 'stop': 10}]) == slice(0, 10)
