import numpy as np
from numba import cuda


# @cuda.jit(device=True)
def color_graph(graph: np.ndarray, order: np.array):
    n = np.uint32(graph.shape[0])
    coloring = np.zeros(n, dtype=np.uint16)
    nc = np.uint16(0)
    for i in order:
        occupied_colors = vertex_neighbour_colors(graph, i, coloring, nc + 2)
        new_color = np.uint16(0)
        for j in range(1, nc + 2):
            if occupied_colors[j] == 0:
                new_color = np.uint16(j)
                break
        if new_color == nc+1:
            nc += 1
        coloring[i] = new_color
        print(f"{occupied_colors=}\t\t{coloring=}")
    return coloring


def vertex_neighbour_colors(graph: np.ndarray, vertex: np.uint32, coloring: np.array, n_colors: np.uint16):
    """Counts number of occurrences of each color of its neighbours.
    Data type conventions:
        - number of vertices = uint32
        - number of colors = uint16

    n_colors may be calculated as max(coloring) + 1
    """
    occurrences = np.zeros(n_colors, dtype=np.uint32)
    for i, edge in enumerate(graph[vertex]):
        if edge == 1:
            occurrences[coloring[i]] += np.uint32(1)
    return occurrences


A = np.array([
    [0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 1, 0, 1, 0],
], dtype=np.int8)
C = np.array([1, 2, 3, 1, 3], dtype=np.uint16)
C1 = np.zeros(5, dtype=np.uint16)
C2 = np.array([1, 9, 0, 0, 0], dtype=np.uint16)
order0 = np.array([3, 1, 4, 0, 2], dtype=np.uint32)
order1 = np.arange(5, dtype=np.uint32)
order2 = np.arange(5, dtype=np.uint32)
np.random.shuffle(order2)

# print(vertex_neighbour_colors(A, 4, C, 3))
print(color_graph(A, order2))
