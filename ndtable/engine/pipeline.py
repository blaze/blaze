from collections import Counter

def toposort(graph):
    """
    Sort the expression graph to resolve the order needed to
    execute operations.
    """
    result = []
    count = Counter()

    for node in graph:
        for child in node:
            count[child] += 1

    sort = [node for node in graph if not count[node]]

    while sort:
        node = sort.pop()
        result.append(node)

        for child in node:
            count[child] -= 1
            if count[child] == 0:
                sort.append(child)

    result.reverse()
    return result

def codegen(graph):
    pass
