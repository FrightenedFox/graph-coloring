import numpy as np
from numba import cuda


@cuda.jit(device=True, nopython=True)
def to_zeros(array: np.ndarray):
    for i in range(len(array)):
        array[i] = 0


@cuda.jit(device=True, nopython=True)
def vertex_neighbour_colors(graph: np.ndarray,
                            vertex: np.uint32,
                            current_coloring: np.ndarray,
                            color_occurrences: np.ndarray):
    """Counts number of occurrences of each color among vertex neighbours.
    Data type conventions:
        - number of vertices = uint32
        - number of colors = uint16

    Parameters
    ----------
    graph :
        Adjacency matrix of the graph made up of zeros and ones.
    current_coloring :
        An array which represents current state of coloring. O means no color is given.
    vertex :
        Vertex which neighbours to search.
    color_occurrences :
        ...
    """
    for neighbour, edge_with_neighbour in enumerate(graph[vertex]):
        if edge_with_neighbour == 1:
            color_occurrences[current_coloring[neighbour]] += 1


@cuda.jit(device=True, nopython=True)
def color_single_graph(graph: np.ndarray,
                       coloring_order: np.ndarray,
                       coloring_out: np.ndarray):
    # TODO: docstrings
    n_colors = np.uint16(0)
    for vertex in coloring_order:
        # TODO: replace exact values with variables (cuda.local.array)
        occupied_colors = cuda.local.array(7, dtype=np.uint32)
        to_zeros(occupied_colors)
        vertex_neighbour_colors(graph, vertex, coloring_out, occupied_colors)
        new_color = 0
        for color in range(1, n_colors + 2):
            if occupied_colors[color] == 0:
                new_color = np.uint16(color)
                break
        if new_color == n_colors + 1:
            n_colors += 1
        coloring_out[vertex] = new_color


@cuda.jit
def color_graphs(graph: np.ndarray,
                 coloring_orders: np.ndarray,
                 colorings_out: np.ndarray):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x

    block_size = cuda.blockDim.x
    grid_size = cuda.gridDim.x

    start = tx + bx * block_size
    stride = block_size * grid_size

    for i in range(start, coloring_orders.shape[0], stride):
        color_single_graph(graph, coloring_orders[i], colorings_out[i])


A = np.array([
    [0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 1, 0, 1, 0],
], dtype=np.int8)
C = np.array([1, 2, 3, 1, 3], dtype=np.uint16)
order0 = np.array([3, 1, 4, 0, 2], dtype=np.uint32)
order1 = np.arange(5, dtype=np.uint32)
order2 = np.arange(5, dtype=np.uint32)
np.random.shuffle(order2)
print(order2)
orders = np.stack((order0, order1, order2))
coloring = np.zeros(A.shape[0], dtype=np.uint16)
colorings = np.zeros_like(orders)

threads_per_block = 1
blocks_per_grid = 30

color_graphs[blocks_per_grid, threads_per_block](A, orders, colorings)
print(colorings)


