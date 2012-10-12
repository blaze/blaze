# Referential
class Categorical(object):
    """
    """

    def __init__(self, mapping, ref):
        self.labels  = set(mapping.keys())
        self.mapping = mapping
        self.ref = ref

    def __contains__(self, other):
        return other in self.labels

    def __getitem__(self, indexer):
        return self.mapping.get(indexer)

    def __repr__(self):
        return 'L(%r)' % list(self.labels)
