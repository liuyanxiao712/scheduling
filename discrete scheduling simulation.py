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


# a0 = Matrix([[0, 0], [0, 0], [0, 0]])
# a1 = Matrix([[0, 1], [0, 0], [0, 0]])
# a2 = Matrix([[0, 0], [0, 1], [0, 0]])
# a3 = Matrix([[0, 0], [0, 0], [0, 1]])
# a4 = Matrix([[1, 0], [0, 0], [0, 1]])
# a5 = Matrix([[1, 0], [0, 0], [0, 0]])
# a6 = Matrix([[0, 1], [1, 0], [0, 0]])
# a7 = Matrix([[0, 0], [1, 0], [0, 0]])
# a8 = Matrix([[0, 0], [0, 1], [1, 0]])
# a9 = Matrix([[0, 0], [0, 0], [1, 0]])
# a10 = Matrix([[1, 1], [0, 0], [0, 0]])
# a11 = Matrix([[0, 0], [1, 1], [0, 0]])
# a12 = Matrix([[0, 0], [0, 0], [1, 1]])


# b0 = Matrix([[1, 1], [1, 0], [0, 0], [0, 1]])
# b1 = Matrix([[0, 1], [1, 0], [0, 0], [0, 1]])
# b2 = Matrix([[1, 1], [0, 0], [0, 0], [0, 1]])
# b3 = Matrix([[1, 0], [1, 0], [0, 0], [0, 1]])
# b4 = Matrix([[1, 1], [1, 0], [0, 0], [0, 0]])
#
# b5 = Matrix([[1, 0], [0, 0], [0, 1], [1, 1]])
# b6 = Matrix([[0, 0], [0, 0], [0, 1], [1, 1]])
# b7 = Matrix([[1, 0], [0, 0], [0, 1], [0, 1]])
# b8 = Matrix([[1, 0], [0, 0], [0, 0], [1, 1]])
# b9 = Matrix([[1, 0], [0, 0], [0, 1], [1, 0]])
#
# b10 = Matrix([[0, 0], [0, 1], [1, 1], [1, 0]])
# b11 = Matrix([[0, 0], [0, 1], [0, 1], [1, 0]])
# b12 = Matrix([[0, 0], [0, 1], [1, 1], [0, 0]])
# b13 = Matrix([[0, 0], [0, 0], [1, 1], [1, 0]])
# b14 = Matrix([[0, 0], [0, 1], [1, 0], [1, 0]])
#
# b15 = Matrix([[0, 1], [1, 1], [1, 0], [0, 0]])
# b16 = Matrix([[0, 1], [0, 1], [1, 0], [0, 0]])
# b17 = Matrix([[0, 1], [1, 1], [0, 0], [0, 0]])
# b18 = Matrix([[0, 0], [1, 1], [1, 0], [0, 0]])
# b19 = Matrix([[0, 1], [1, 0], [1, 0], [0, 0]])
#
# b20 = Matrix([[1, 1], [0, 0], [0, 0], [0, 0]])
# b21 = Matrix([[1, 0], [0, 0], [0, 1], [0, 0]])
# b22 = Matrix([[1, 0], [0, 0], [0, 0], [0, 1]])
# b23 = Matrix([[0, 1], [1, 0], [0, 0], [0, 0]])
# b24 = Matrix([[0, 0], [1, 1], [0, 0], [0, 0]])
# b25 = Matrix([[0, 0], [1, 0], [0, 0], [0, 1]])
# b26 = Matrix([[0, 1], [0, 0], [1, 0], [0, 0]])
# b27 = Matrix([[0, 0], [0, 1], [1, 0], [0, 0]])
# b28 = Matrix([[0, 0], [0, 0], [1, 1], [0, 0]])
# b29 = Matrix([[0, 0], [0, 1], [0, 0], [1, 0]])
# b30 = Matrix([[0, 0], [0, 0], [0, 1], [1, 0]])
# b31 = Matrix([[0, 0], [0, 0], [0, 0], [1, 1]])


a0 = Matrix([[0], [0], [0]])
a1 = Matrix([[1], [0], [0]])
a2 = Matrix([[0], [1], [0]])
a3 = Matrix([[0], [0], [1]])

schedule_list = [a0, a1, a2, a3]
# schedule_list = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12]
# schedule_list2 = [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25, b26, b27, b28, b29, b30, b31]

list1 = [[a0, 0], [a1, 1], [a2, 2], [a3, 3]]
# list for triangle network
# list1 = [[a0, 0], [a1, 1], [a2, 2], [a3, 3], [a4, 4], [a5, 5], [a6, 6], [a7, 7], [a8, 8], [a9, 9], [a10, 10], [a11, 11], [a12, 12]]
# list for 4-link line network
# list2 = [[b0, 0], [b1, 1], [b2, 2], [b3, 3], [b4, 4], [b5, 5], [b6, 6], [b7, 7], [b8, 8], [b9, 9], [b10, 10], [b11, 11], [b12, 12], [b13, 13], [b14, 14], [b15, 15], [b16, 16], [b17, 17], [b18, 18], [b19, 19], [b20, 20], [b21, 21], [b22, 22], [b23, 23], [b24, 24], [b25, 25], [b26, 26], [b27, 27], [b28, 28], [b29, 29], [b30, 30], [b31, 31]]

def rate(l):
    # input a list of indexes and return a rate vector(numpy array)
    rate_vec = np.array([0, 0, 0])
    num = len(l)  # number of schedules
    for w in range(num):
        for j in range(3):
            rate_vec[j] += (schedule_list[l[w]][j,0] + schedule_list[l[w]][j,1])
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

def dfs(graph, start, end):
    #  Input a graph dict and output list of all cycles
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

def plot_points(array):
    x = array[:, 0]
    y = array[:, 1]
    z = array[:, 2]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    T = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    u = T[:, 0]
    v = T[:, 1]
    w = T[:, 2]
    ax.plot_trisurf(u, v, w, alpha=0.4, linewidth=0.2, antialiased=True, color='b')
    plt.show()


def plot_surface(array):
    fig = plt.figure()
    x = array[:, 0]
    y = array[:, 1]
    z = array[:, 2]
    plt.title('rate region')
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, alpha =0.5, linewidth=0.2, antialiased=True, color = 'r', edgecolor='b')
    plt.show()


if __name__ == "__main__":

    cycles_list = schedule_graph_library.cyclelist_tri_m1e1
    rate_all = rate_tri_m1e1(cycles_list[0])
    for i in range(1, len(cycles_list)):
        c = cycles_list[i]
        r = rate_tri_m1e1(c)
        rate_all = np.vstack((rate_all, r))
        # if r[0] + r[1] + r[2] >= 1:
        #     print(r, cycles_list[i])
        #     rate_all = np.vstack((rate_all, r))
    # rate_all = np.vstack((rate_all, np.array([0,0,0])))
        # rate_all = np.vstack((rate_all, r))
    # print(rate_all)
    plot_points(rate_all)
    plot_surface(rate_all)



