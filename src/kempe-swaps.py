from typing import List, Tuple

import numpy as np
from numba import njit


def find_kempe_chain(graph: np.ndarray, coloring: np.ndarray) -> Tuple[List[List[int]], List[np.ndarray]]:
    """Find all unique Kempe chains.

    Parameters
    ----------
    graph :
        Adjacency matrix of the graph made up of zeros and ones.
    coloring :
        Valid coloring of the graph. List of color numbers 1:n.

    Returns
    -------
    Kempe_chains :
        List of Kempe chains
    Kempe_chains_colors :
        Colors of each Kempe chain
    """
    kempe_chains, kempe_chains_colors = [], []
    for i, row in enumerate(graph):
        a_color = coloring[i]
        i_neighbours = np.where(row[i + 1:])[0] + (i + 1)  # i + 1 not to add the same pair twice
        for neighbour in i_neighbours:
            b_color = coloring[neighbour]
            chain = list(np.sort(recursive_chain(graph, coloring, neighbour, [], [a_color, b_color])))
            if chain not in kempe_chains:
                kempe_chains.append(chain)
                kempe_chains_colors.append([a_color, b_color])
    return kempe_chains, np.array(kempe_chains_colors)


def recursive_chain(graph: np.ndarray,
                    coloring: np.ndarray,
                    node: int,
                    kempe_chain: List[int],
                    kempe_colors: List[int]) -> List[int]:
    """Recursively find Kempe chain, starting from given node (vertex).

    Parameters
    ----------
    graph :
        Adjacency matrix of the graph made up of zeros and ones.
    coloring :
        Valid coloring of the graph. List of color numbers 1:n.
    node :
        Node (vertex) to start searching from.
    kempe_chain :
        Array, where future Kempe chain will be held. Just give an empty list.
    kempe_colors :
        Colors to look for.

    Returns
    -------
        Found Kempe chain.
    """
    kempe_chain.append(node)
    node_neighbours = np.where(graph[node])[0]
    for neighbour in node_neighbours:
        if neighbour not in kempe_chain and coloring[neighbour] in kempe_colors:
            recursive_chain(graph, coloring, neighbour, kempe_chain, kempe_colors)
    return kempe_chain


@njit
def swap_kempe_chain_colors(coloring: np.ndarray,
                            kempe_chain: np.ndarray,
                            colors: np.ndarray):
    """Swaps colors in the coloring accordingly to Kempe chain.

    Parameters
    ----------
    coloring :
            Valid coloring of the graph. List of color numbers 1:n,
            which will be modified and used as an output array as well,
            so it's better to give a copy of the original coloring.
    kempe_chain :
        Kempe chain - list of nodes, that must be swapped.
    colors :
        Colors of the given Kempe chain.
    """
    a_col, b_col = colors
    for link in kempe_chain:
        if coloring[link] == a_col:
            coloring[link] = b_col
        else:
            coloring[link] = a_col
