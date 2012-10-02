from idx import Index, AutoIndex

class NDTable(object):
    def __init__(self, obj, datashape, index=None, metadata=None):
        self.datashape = datashape
        self.metadata = metadata

        if isinstance(obj, list):
            self.index = AutoIndex(*obj)
        elif isinstance(obj, Index):
            self.index = index

    def __getitem__(self, indexer):
        pass

    def __getslice__(self, i, j):
        pass

    def from_sql(dburl, query):
        pass

    def from_csv(fname, *params):
        pass

    # IPython notebook integration
    def to_html(self):
        return '<table></table>'

    def _repr_html_(self):
        return ('<div style="max-height:1000px;'
                'max-width:1500px;overflow:auto;">\n' +
                self.to_html() + '\n</div>')
