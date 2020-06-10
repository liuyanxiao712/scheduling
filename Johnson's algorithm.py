from sympy import symbols, Matrix
import string
from sympy.matrices import randMatrix, zeros
import random
from itertools import combinations, permutations
import numpy as np
import gc
import multiprocessing
from copy import deepcopy
from collections import defaultdict



b0 = Matrix([[1, 1], [1, 0], [0, 0], [0, 1]])
b1 = Matrix([[0, 1], [1, 0], [0, 0], [0, 1]])
b2 = Matrix([[1, 1], [0, 0], [0, 0], [0, 1]])
b3 = Matrix([[1, 0], [1, 0], [0, 0], [0, 1]])
b4 = Matrix([[1, 1], [1, 0], [0, 0], [0, 0]])

b5 = Matrix([[1, 0], [0, 0], [0, 1], [1, 1]])
b6 = Matrix([[0, 0], [0, 0], [0, 1], [1, 1]])
b7 = Matrix([[1, 0], [0, 0], [0, 1], [0, 1]])
b8 = Matrix([[1, 0], [0, 0], [0, 0], [1, 1]])
b9 = Matrix([[1, 0], [0, 0], [0, 1], [1, 0]])

b10 = Matrix([[0, 0], [0, 1], [1, 1], [1, 0]])
b11 = Matrix([[0, 0], [0, 1], [0, 1], [1, 0]])
b12 = Matrix([[0, 0], [0, 1], [1, 1], [0, 0]])
b13 = Matrix([[0, 0], [0, 0], [1, 1], [1, 0]])
b14 = Matrix([[0, 0], [0, 1], [1, 0], [1, 0]])

b15 = Matrix([[0, 1], [1, 1], [1, 0], [0, 0]])
b16 = Matrix([[0, 1], [0, 1], [1, 0], [0, 0]])
b17 = Matrix([[0, 1], [1, 1], [0, 0], [0, 0]])
b18 = Matrix([[0, 0], [1, 1], [1, 0], [0, 0]])
b19 = Matrix([[0, 1], [1, 0], [1, 0], [0, 0]])

b20 = Matrix([[1, 1], [0, 0], [0, 0], [0, 0]])
b21 = Matrix([[1, 0], [0, 0], [0, 1], [0, 0]])
b22 = Matrix([[1, 0], [0, 0], [0, 0], [0, 1]])
b23 = Matrix([[0, 1], [1, 0], [0, 0], [0, 0]])
b24 = Matrix([[0, 0], [1, 1], [0, 0], [0, 0]])
b25 = Matrix([[0, 0], [1, 0], [0, 0], [0, 1]])
b26 = Matrix([[0, 1], [0, 0], [1, 0], [0, 0]])
b27 = Matrix([[0, 0], [0, 1], [1, 0], [0, 0]])
b28 = Matrix([[0, 0], [0, 0], [1, 1], [0, 0]])
b29 = Matrix([[0, 0], [0, 1], [0, 0], [1, 0]])
b30 = Matrix([[0, 0], [0, 0], [0, 1], [1, 0]])
b31 = Matrix([[0, 0], [0, 0], [0, 0], [1, 1]])

schedule_list2 = [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25, b26, b27, b28, b29, b30, b31]

list2 = [[b0, 0], [b1, 1], [b2, 2], [b3, 3], [b4, 4], [b5, 5], [b6, 6], [b7, 7], [b8, 8], [b9, 9], [b10, 10], [b11, 11], [b12, 12], [b13, 13], [b14, 14], [b15, 15], [b16, 16], [b17, 17], [b18, 18], [b19, 19], [b20, 20], [b21, 21], [b22, 22], [b23, 23], [b24, 24], [b25, 25], [b26, 26], [b27, 27], [b28, 28], [b29, 29], [b30, 30], [b31, 31]]


# def simple_cycles(start_vertex):
def simple_cycles(G):
    # Yield every elementary cycle in python graph G exactly once
    # Expects a dictionary mapping from vertices to iterables of vertices
    # G = graph
    def _unblock(thisnode, blocked, B):
        stack = set([thisnode])
        while stack:
            node = stack.pop()
            if node in blocked:
                blocked.remove(node)
                stack.update(B[node])
                B[node].clear()
    G = {v: set(nbrs) for (v, nbrs) in G.items()}  # make a copy of the graph
    sccs = strongly_connected_components(G)
    while sccs:
        scc = sccs.pop()
        # startnode = scc.pop()
        startnode = scc[current_phase]
        path = [startnode]
        blocked = set()
        closed = set()
        blocked.add(startnode)
        B = defaultdict(set)
        stack = [(startnode, list(G[startnode]))]
        while stack:
            thisnode, nbrs = stack[-1]
            if nbrs:
                nextnode = nbrs.pop()
                if nextnode == startnode:
                    yield path[:]
                    closed.update(path)
                    # print(path)
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
        # H = subgraph(G, set(scc))
        # print("H", H)
        # sccs.extend(strongly_connected_components(H))
        # break



def draw_graph(l, s):
    """
    :param l: A list including [schedule matrix, index] and the size of this square matrix
    :param s: schedule list
    :param size: size of graph matrix
    :return: matrix denoting schedule graph
    """
    graph_m = zeros(32)
    for u in range(32):
        for v in range(32):
            c = Matrix([[0, 0], [0, 0], [0, 0], [0, 0]])
            c[:, 0] = l[u][0][:, 1]
            c[:, 1] = l[v][0][:, 0]
            if c in s:
                graph_m[u, v] = 1
            del c
    gc.collect()
    return graph_m


def graph_2_dict(graph):
    #  change denoting a graph from matrix to dict
    g = {}
    size = graph.rows
    for p in range(size):
        temp = []
        for q in range(size):
            if graph[p, q] == 1:
                temp.append(q)
        g[p] = temp
        del temp
    del size
    gc.collect()
    return g


def dfs(graph, start, end):
    fringe = [(start, [])]
    while fringe:
        state, path = fringe.pop()
        if path and state == end:
            yield path
            continue
        for next_state in graph[state]:
            if next_state in path:
                continue
            fringe.append((next_state, path+[next_state]))
        del state
        del path
    gc.collect()


def strongly_connected_components(graph):
    # Tarjan's algorithm for finding SCC's
    # Robert Tarjan. "Depth-first search and linear graph algorithms." SIAM journal on computing. 1972.
    # Code by Dries Verdegem, November 2012
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
    # H = subgraph(G, set(scc))
    return {v: G[v] & vertices for v in vertices}


def rate(l):
    # input a list of indexes and return a rate vector(numpy array)
    rate_vec = np.array([0, 0, 0, 0])
    num = len(l)  # number of schedules
    for w in range(num):
        for j in range(4):
            rate_vec[j] += (schedule_list2[l[w]][j,0] + schedule_list2[l[w]][j,1])
    rate_vec = rate_vec / (2 * num)
    rate_vec = rate_vec.tolist()
    for v in range(4):
        rate_vec[v] = round(rate_vec[v], 3)
    return rate_vec


graph = {0: [6, 10, 11, 12, 13, 14, 16, 26, 27, 28, 29, 30, 31], 1: [6, 10, 11, 12, 13, 14, 16, 26, 27, 28, 29, 30, 31], 2: [6, 10, 11, 12, 13, 14, 16, 26, 27, 28, 29, 30, 31], 3: [1, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31], 4: [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 20, 21, 22, 26, 27, 28, 29, 30, 31], 5: [1, 12, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28], 6: [1, 12, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28], 7: [1, 12, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28], 8: [1, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31], 9: [0, 1, 2, 3, 4, 7, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], 10: [0, 1, 2, 3, 4, 7, 17, 20, 21, 22, 23, 24, 25], 11: [0, 1, 2, 3, 4, 7, 17, 20, 21, 22, 23, 24, 25], 12: [0, 1, 2, 3, 4, 7, 17, 20, 21, 22, 23, 24, 25], 13: [0, 1, 2, 3, 4, 7, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], 14: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 17, 20, 21, 22, 23, 24, 25, 29, 30, 31], 15: [2, 5, 6, 7, 8, 9, 11, 20, 21, 22, 29, 30, 31], 16: [2, 5, 6, 7, 8, 9, 11, 20, 21, 22, 29, 30, 31], 17: [2, 5, 6, 7, 8, 9, 11, 20, 21, 22, 29, 30, 31], 18: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 17, 20, 21, 22, 23, 24, 25, 29, 30, 31], 19: [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 20, 21, 22, 26, 27, 28, 29, 30, 31], 20: [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 20, 21, 22, 26, 27, 28, 29, 30, 31], 21: [0, 1, 2, 3, 4, 7, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], 22: [1, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31], 23: [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 20, 21, 22, 26, 27, 28, 29, 30, 31], 24: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 17, 20, 21, 22, 23, 24, 25, 29, 30, 31], 25: [1, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31], 26: [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 20, 21, 22, 26, 27, 28, 29, 30, 31], 27: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 17, 20, 21, 22, 23, 24, 25, 29, 30, 31], 28: [0, 1, 2, 3, 4, 7, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], 29: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 17, 20, 21, 22, 23, 24, 25, 29, 30, 31], 30: [0, 1, 2, 3, 4, 7, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], 31: [1, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31]}
l = []
count = 0
check = -1
current_phase = 31
for u in simple_cycles(graph):
    # print(u)
    if count == 0:
        l.append(rate(u))
        count += 1
        continue
    u = rate(u)
    if u not in l:
        l.append(u)
        count += 1
    if count >= 200:
        count = 1
        filename = "Johnson_phase" + str(current_phase) + ".txt"
        f = open(filename, 'a')
        f.write(str(l))
        del l
        l = []
print("Game over for phase", current_phase)