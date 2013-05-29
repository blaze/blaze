import hashlib
from collections import Iterable

KEY = '%s:%s'

#------------------------------------------------------------------------
# Hash Ring
#------------------------------------------------------------------------

class Ring(Iterable):

    def __init__(self, nodes, replicas=3):
        nodes = nodes or []

        self.replicas = replicas
        self._keys = []
        self.cluster = dict()

        for node in nodes:
            self.add_node(node)

    # --------------------------------------------

    def add_node(self, node):
        for i in range(self.replicas):
            key = self.hash(KEY % (node, i))
            self.cluster[key] = node
            self._keys.append(key)

        self._keys.sort()

    # --------------------------------------------

    def get_node(self, cluster_key):
        return self.get_node_idx(cluster_key)[0]

    # --------------------------------------------

    def get_node_idx(self, cluster_key):
        if not self.cluster:
            return None, None

        key = self.hash(cluster_key)

        nodes = self._keys
        for i, node in enumerate(nodes):
            node = nodes[i]
            if key <= node:
                return self.cluster[node], i

        return self.cluster[nodes[0]], 0

    def get_nodes(self, cluster_key):
        if not self.cluster:
            yield None, None

        node, idx = self.get_node_idx(cluster_key)

        for key in self._keys[idx:]:
            yield self.cluster[key]

        while True:
            for key in self._keys:
                yield self.cluster[key]

    # --------------------------------------------

    def remove_node(self, node):
        for i in range(self.replicas):
            key = self.hash(KEY % (node, i))
            del self.cluster[key]
            self._keys.remove(key)

    # --------------------------------------------

    def hash(self, key):
        m = hashlib.sha1()
        m.update(key)
        return int(m.hexdigest(), 16)

    def __contains__(self, node):
        return node in self.cluster

    def __iter__(self):
        return iter(self.cluster.values())

    def __str__(self):
        return '<Ring [%s]>' % ','.join(str(s) for s in self.cluster.items())

if __name__ == '__main__':
    nodes = ['localhost:5000',
             'localhost:5001',
             'localhost:5002']

    ring = Ring(nodes, replicas=3)
    server = ring.get_node('foo')
    print server
