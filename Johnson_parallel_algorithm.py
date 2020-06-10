# A dependency-free version of networkx's implementation of Johnson's cycle finding algorithm
# Original implementation: https://github.com/networkx/networkx/blob/master/networkx/algorithms/cycles.py#L109
# Original paper: Donald B Johnson. "Finding all the elementary circuits of a directed graph." SIAM Journal on Computing. 1975.


from collections import defaultdict
import multiprocessing
from copy import deepcopy


# def simple_cycles(G, target):
def simple_cycles(vertex):
    """
    :param G: expect a dictionary mapping from vertices to iterables of vertices. NOTE: when perform paralle, I delete G
    :param vertex: since we use parallel computing, each target corresponds to a node in a SCC
    :return: Yield every elementary cycle in python graph G exactly once
    """
    cycles = []
    G = graph
    def _unblock(thisnode, blocked, B):
        stack = set([thisnode])
        while stack:
            node = stack.pop()
            if node in blocked:
                blocked.remove(node)
                stack.update(B[node])
                B[node].clear()
    G = {v: set(nbrs) for (v, nbrs) in G.items()}  # make a copy of the graph
    sccs = strongly_connected_components(G)  # return strongly connected components
    # print("scc", sccs)
    scc = sccs[0]  # a list of one SCC, in our case since there's only one SCC, hee scc is just our whole graph
    # startnode = scc.pop()
    startnode = scc[vertex]
    path = [startnode]
    blocked = set()
    closed = set()
    blocked.add(startnode)
    B = defaultdict(set)
    stack = [(startnode, list(G[startnode]))]
    print("stack", stack)
    while stack:
        thisnode, nbrs = stack[-1]
        if nbrs:
            nextnode = nbrs.pop()
            if nextnode == startnode:
                # yield path[:]
                cycles.append(deepcopy(path))
                closed.update(path)
            elif nextnode not in blocked:
                path.append(nextnode)
                stack.append((nextnode, list(G[nextnode])))
                closed.discard(nextnode)
                blocked.add(nextnode)
                continue
        if not nbrs:
            if thisnode in closed:
                _unblock(thisnode, blocked, B)
            else:
                for nbr in G[thisnode]:
                    if thisnode not in B[nbr]:
                        B[nbr].add(thisnode)
            stack.pop()
            path.pop()
    remove_node(G, startnode)
    print(G)
    print(scc)
    H = subgraph(G, set(scc))
    sccs.extend(strongly_connected_components(H))
    print(cycles)
    return cycles

def strongly_connected_components(graph):
    # Tarjan's algorithm for finding SCC's
    # return a list of lists, denoting SCC's
    # Downloaded from http://www.logarithmic.net/pfh/blog/01208083168

    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    result = []

    def _strong_connect(node):
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)

        successors = graph[node]
        for successor in successors:
            if successor not in index:
                _strong_connect(successor)
                lowlink[node] = min(lowlink[node], lowlink[successor])
            elif successor in stack:
                lowlink[node] = min(lowlink[node], index[successor])
        if lowlink[node] == index[node]:
            connected_component = []
            while True:
                successor = stack.pop()
                connected_component.append(successor)
                if successor == node: break
            result.append(connected_component[:])
    for node in graph:
        if node not in index:
            _strong_connect(node)
    return result


def remove_node(G, target):
    # Completely remove a node from the graph
    # Expects values of G to be sets
    del G[target]
    for nbrs in G.values():
        nbrs.discard(target)


def subgraph(G, vertices):
    # Get the subgraph of G induced by set vertices
    # Expect values of G to be sets
    return {v: G[v] & vertices for v in vertices}

# print(subgraph({0: {3, 5, 7}, 1: set(), 3: {0, 5}, 4: {8, 6}, 5: {0, 3, 7}, 6: {8, 4}, 7: {0, 8, 5}, 8: {4, 6, 7}}, [1, 2, 6, 4, 8, 7, 5, 3, 0]))
graph = {0: [7, 3, 5], 1: [2], 2: [7, 1], 3: [0, 5], 4: [6, 8], 5: [0, 3, 7], 6: [4, 8], 7: [0, 2, 5, 8], 8: [4, 6, 7]}
print(simple_cycles(1))

# p = multiprocessing.Pool(4)
# b = p.map(simple_cycles, [i for i in range(9)])
# p.close()
# p.join()
# print(b)
# print(tuple(simple_cycles(graph)))