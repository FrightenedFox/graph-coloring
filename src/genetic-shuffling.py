import numpy as np
from numba import jit


def generate_random_orderings(n_elements: int, k_orderings: int, rng: np.random.Generator) -> np.ndarray:
    """ Generate k orderings of vectors with n elements.

    Parameters
    ----------
    n_elements :
        Number of elements in each ordering.
    k_orderings
        Number of orderings.
    rng
        numpy.random.Generator

    Returns
    -------
        numpy.ndarray of size (n, k) - k orderings with n elements each

    """
    return rng.permuted(np.tile(np.arange(n_elements), (k_orderings, 1)), axis=1)


def mutate_orderings(orderings: np.ndarray, probability: float, rng: np.random.Generator, inplace=True) -> np.ndarray:
    if not inplace:
        orderings = np.array(orderings)
    k_orderings, n_elements = orderings.shape
    m_mutations = np.ceil(orderings.size * probability / 2).astype(int)     # because swaps double n of mutations
    rows = np.sort(rng.integers(k_orderings, size=m_mutations))
    origins, destinations = np.zeros(m_mutations, dtype=int), np.zeros(m_mutations, dtype=int)
    pos_id = 0
    for row in np.unique(rows):
        half_elements = np.ceil(n_elements / 2).astype(int)
        n_swaps = np.min([np.sum(rows == row), half_elements])   # min to eliminate occasions when n_swaps > n_elements
        addresses = rng.choice(n_elements, size=n_swaps * 2, replace=False)
        origins[pos_id:(pos_id + n_swaps)] = addresses[:n_swaps]
        destinations[pos_id:(pos_id + n_swaps)] = addresses[n_swaps:]
        pos_id += n_swaps
    mask = np.where(origins == destinations)[0]
    rows, origins, destinations = np.delete(rows, mask), np.delete(origins, mask), np.delete(destinations, mask)
    orderings[rows, origins], orderings[rows, destinations] = orderings[rows, destinations], orderings[rows, origins]
    return orderings
