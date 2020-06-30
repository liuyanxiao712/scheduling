from sympy import symbols, Matrix
import string
from sympy.matrices import randMatrix, zeros
import random
from itertools import combinations, permutations
import numpy as np
import gc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import schedule_graph_library
# vertices of (M1, E1) schedule graph of triangle example
a0 = Matrix([[0], [0], [0]])
a1 = Matrix([[1], [0], [0]])
a2 = Matrix([[0], [1], [0]])
a3 = Matrix([[0], [0], [1]])
# graph_tri_M1E1 = Matrix([[1,1,1,1], [1,1,0, 1], [1,1,1, 0], [1,0,1,1]])
# schedule_list = [a0, a1, a2, a3]

# vertices of (M1, E1) schedule graph of 4-link example
b0 = Matrix([[0], [0], [0], [0]])
b1 = Matrix([[1], [0], [0], [0]])
b2 = Matrix([[0], [1], [0], [0]])
b3 = Matrix([[0], [0], [1], [0]])
b4 = Matrix([[0], [0], [0], [1]])
b5 = Matrix([[1], [1], [0], [0]])
b6 = Matrix([[1], [0], [0], [1]])
b7 = Matrix([[0], [1], [1], [0]])
b8 = Matrix([[0], [0], [1], [1]])
graph_4link_M1E1 = Matrix([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 0, 1, 1, 0, 1, 0, 1],
                           [1, 1, 1, 0, 1, 1, 1, 0, 0],

                           [1, 1, 1, 1, 0, 1, 0, 1, 0],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 0, 0, 1, 0, 1, 0, 0],

                           [1, 1, 0, 1, 1, 0, 1, 0, 1],
                           [1, 1, 1, 0, 0, 1, 0, 0, 0],
                           [1, 1, 1, 1, 0, 1, 0, 1, 0]])
graph_dict = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [0, 1, 3, 4, 6, 8], 2: [0, 1, 2, 4, 5, 6], 3: [0, 1, 2, 3, 5, 7], 4: [0, 1, 2, 3, 4, 5, 6, 7, 8], 5: [0, 1, 4, 6], 6: [0, 1, 3, 4, 6, 8], 7: [0, 1, 2, 5], 8: [0, 1, 2, 3, 5, 7]}
# list for triangle network
# list1 = [[a0, 0], [a1, 1], [a2, 2], [a3, 3], [a4, 4], [a5, 5], [a6, 6], [a7, 7], [a8, 8], [a9, 9], [a10, 10], [a11, 11], [a12, 12]]

def draw_graph(l, s, size):
    """
    :param l: A list including [schedule matrix, index] and the size of this square matrix
    :param s: schedule list
    :param size: size of graph matrix
    :return: matrix denoting schedule graph
    """
    graph_m = zeros(size)
    for u in range(size):
        for v in range(size):
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

# def dfs(graph, start, end):
#     #  Input a graph dict and output list of all cycles
#     fringe = [(start, [])]
#     while fringe:
#         state, path = fringe.pop()
#         if path and state == end:
#             yield path
#             continue
#         for next_state in graph[state]:
#             if next_state in path:
#                 continue
#             fringe.append((next_state, path+[next_state]))
#         del state
#         del path
#     gc.collect()
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

def rate(l):
    # input a list of indexes and return a rate vector(numpy array)
    rate_vec = np.array([0, 0, 0])
    num = len(l)  # number of schedules
    for w in range(num):
        for j in range(3):
            rate_vec[j] += (schedule_list[l[w]][j, 0] + schedule_list[l[w]][j, 1])
    rate_vec = rate_vec / (2 * num)
    return rate_vec

def rate_tri_m1e1(l):
    # input a list of indexes and return a rate vector(numpy array)
    rate_vec = np.array([0, 0, 0])
    num = len(l)  # number of schedules
    for w in range(num):
        if l[w] == 1:
            rate_vec[0] += 1
        elif l[w] == 2:
            rate_vec[1] += 1
        elif l[w] == 3:
            rate_vec[2] += 1
    rate_vec = rate_vec / num
    return rate_vec

def rate_4link_m1e1(l):
    # input a list of indexes and return a rate vector(numpy array)
    rate_vec = np.array([0, 0, 0, 0])
    num = len(l)  # number of schedules(num of elements in this cycle)
    for w in range(num):
        if l[w] == 1:
            rate_vec[0] += 1
        elif l[w] == 2:
            rate_vec[1] += 1
        elif l[w] == 3:
            rate_vec[2] += 1
        elif l[w] == 4:
            rate_vec[3] += 1
        elif l[w] == 5:
            rate_vec[0] += 1
            rate_vec[1] += 1
        elif l[w] == 6:
            rate_vec[0] += 1
            rate_vec[3] += 1
        elif l[w] == 7:
            rate_vec[1] += 1
            rate_vec[2] += 1
        elif l[w] == 8:
            rate_vec[2] += 1
            rate_vec[3] += 1
    rate_vec = rate_vec / num
    return rate_vec

def plot_points(array):
    x = array[:, 0]
    y = array[:, 1]
    z = array[:, 2]
    f = plt.figure()
    ax = Axes3D(f)
    ax.scatter(x, y, z)

    # convex hull for (1 0 0)(0 1 0)(0 0 1)
    # T = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # u = T[:, 0]
    # v = T[:, 1]
    # w = T[:, 2]
    # ax.plot_trisurf(u, v, w, alpha=0.4, linewidth=0.2, antialiased=True, color='r')
    # T = np.array([[1, 0, 0], [0, 0, 0], [0, 0.000000000001, 1]])
    # u = T[:, 0]
    # v = T[:, 1]
    # w = T[:, 2]
    # ax.plot_trisurf(u, v, w, alpha=0.4, linewidth=0.2, antialiased=True, color='r')

    # convex hull for (1 0 0)(0 1 0)(0 0 1)(1 0 1)
    # T = np.array([[1, 0.00000000001, 1], [1, 0, 0], [0, 1, 0]])
    # u = T[:, 0]
    # v = T[:, 1]
    # w = T[:, 2]
    # ax.plot_trisurf(u, v, w, alpha=0.4, linewidth=0.2, antialiased=True, color='r')
    # T = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    # u = T[:, 0]
    # v = T[:, 1]
    # w = T[:, 2]
    # ax.plot_trisurf(u, v, w, alpha=0.4, linewidth=0.2, antialiased=True, color='r')
    # T = np.array([[1, 0.000000000001, 1], [1, 0, 0], [0, 0, 1]])
    # u = T[:, 0]
    # v = T[:, 1]
    # w = T[:, 2]
    # ax.plot_trisurf(u, v, w, alpha=0.4, linewidth=0.2, antialiased=True, color='r')
    # T = np.array([[1, 0, 0], [0, 0, 0], [0, 0.000000000001, 1]])
    # u = T[:, 0]
    # v = T[:, 1]
    # w = T[:, 2]
    # ax.plot_trisurf(u, v, w, alpha=0.4, linewidth=0.2, antialiased=True, color='r')

    # plot our rate region
    # T = np.array([[1, 0, 0], [0, 0, 1], [0.5, 0.5, 0.5]])
    # u = T[:, 0]
    # v = T[:, 1]
    # w = T[:, 2]
    # ax.plot_trisurf(u, v, w, alpha=0.4, linewidth=0.2, antialiased=True, color='b')
    # T = np.array([[1, 0, 0], [0, 1, 0], [0.500000000000000001, 0.50000000002, 0.5]])
    # u = T[:, 0]
    # v = T[:, 1]
    # w = T[:, 2]
    # ax.plot_trisurf(u, v, w, alpha=0.4, linewidth=0.2, antialiased=True, color='b')
    # T = np.array([[0, 1, 0], [0, 0, 1], [0.5, 0.50000000000000002, 0.500000000000000001]])
    # u = T[:, 0]
    # v = T[:, 1]
    # w = T[:, 2]
    # ax.plot_trisurf(u, v, w, alpha=0.4, linewidth=0.2, antialiased=True, color='b')

    plt.show()

def plot_surface(array):
    fig = plt.figure()
    x = array[:, 0]
    y = array[:, 1]
    z = array[:, 2]
    plt.title('rate region')
    ax = fig.gca(projection='3d')
    # ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, z, alpha =0.5, linewidth=0.2, antialiased=True, color = 'b', edgecolor='g')
    plt.show()

if __name__ == "__main__":
    # cycles = schedule_graph_library.cyclelist_4link_m1e1
    # rate_all = rate_cellular(cycles[0])
    # for i in range(55268):
    #     rate_all = np.vstack((rate_all, rate_cellular(cycles[i])))
    # data = rate_all[:,[1,2,3]]
    # plot_points(data)