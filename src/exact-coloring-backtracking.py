import numpy as np
from numba import jit
from numba.typed import List


@jit(nopython=True)
def coloring(graph: np.ndarray,
             current_coloring: np.ndarray,
             vertex: np.uint32,
             vertex_color: np.uint16,
             n_colors: np.uint16,
             colorings_out: List):
    """ Color graph using backtracking and recurrence

    Parameters
    ----------
    graph :
        Adjacency matrix of the graph made up of zeros and ones.
    current_coloring :
        An array which represents current state of coloring. O means no color is given.
    vertex :
        Current vertex of the graph.
    vertex_color :
        Current vertex color.
    n_colors :
        How many colors to color graph with.
    colorings_out :
        An output array of solutions of type `numba.typed.List`.
    """
    current_coloring[vertex] = vertex_color
    vertex += 1
    if vertex < graph.shape[0]:
        occupied_colors = np.zeros(n_colors + 1, dtype=np.uint32)
        for neighbour, edge_with_neighbour in enumerate(graph[vertex]):
            if edge_with_neighbour == 1:
                occupied_colors[current_coloring[neighbour]] += 1
        unoccupied_colors = np.where(occupied_colors[1:] == 0)[0] + 1   # workaround to skip zeroth color counting
        for color in unoccupied_colors:
            coloring(graph, current_coloring.copy(), vertex, color, n_colors, colorings_out)
    else:
        colorings_out.append(current_coloring)


A = np.array([
    [0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 1, 0, 1, 0],
], dtype=np.int8)
# colorings = np.zeros((10, 5), dtype=np.uint16)
colorings = List()
colorings.append(np.zeros(5, dtype=np.uint16))
colorings.pop()
coloring(
    graph=A,
    current_coloring=np.zeros(5, dtype=np.uint16),
    vertex=0,
    vertex_color=1,
    n_colors=3,
    colorings_out=colorings
)

print(np.stack(colorings))

