from typing import Union, List

import numpy as np
import pandas as pd
from numba import cuda, njit
from numba.typed import List as NumbaList

import exact_coloring_backtracking as ecb
import genetic_shuffling as gs
import greedy_coloring_cuda as gcc
import kempe_swaps as ks


@njit
def cost_func_1(graph: np.ndarray,
                group_sizes: np.ndarray,
                coloring: np.ndarray):
    output = 0
    for i in range(1, max(coloring) + 1):
        group_i = np.where(coloring == i)[0]  # get indexes of groups with color i
        for j, u in enumerate(group_i):
            for v in group_i[j + 1:]:  # j + 1 to escape counting same group twice
                output += (group_sizes[u] + group_sizes[v]) * graph[u, v]
    return output


@njit
def cost_func_2(group_sizes: np.ndarray,
                coloring: np.ndarray):
    output = 0
    avg_n_of_people_per_table = group_sizes.sum() / max(coloring)
    for i in range(1, max(coloring) + 1):
        group_i = np.where(coloring == i)[0]  # get indexes of groups with color i
        output += np.abs(group_sizes[group_i].sum() - avg_n_of_people_per_table)
    return output


def print_colorings(colorings: np.ndarray, labels: Union[List[str], None] = None):
    """Function to print colorings"""
    if len(colorings) > 10:
        colorings = colorings[:10]
        print("Printing first 10 colorings:")
    if labels:
        labels = labels[:len(colorings)]
        for label, coloring in zip(labels, colorings):
            print(f"{label}:\t{coloring}")
    else:
        for i, coloring in enumerate(colorings):
            print(f"{i}:\t{coloring}")
    print()


def print_kempe_chains(kempe_chains: dict):
    """Function to print colorings"""
    for i, chain in kempe_chains.items():
        print(f"{i}:\t{chain}")
        if i >= 10:
            break
    print()


def prepare_smallest_last_ordering(graph: np.ndarray):
    graph_df = pd.DataFrame(graph)
    ordering = np.zeros(len(graph), dtype=np.int32)
    for i in range(len(graph)):
        ordering[i] = graph_df.sum().sort_values().index[0]
        graph_df.drop(ordering[i], axis=1, inplace=True)
    return ordering


def naive_orderings(strict_graph: np.ndarray):
    # Generate orderings
    degrees = strict_graph.sum(axis=1)
    largest_first_ordering = np.argsort(degrees)[::-1]
    smallest_last_ordering = prepare_smallest_last_ordering(strict_graph)

    coloring_orders = np.stack((largest_first_ordering, smallest_last_ordering))

    # Transfer data to GPU
    strict_graph_dev = cuda.to_device(strict_graph)
    coloring_orders_dev = cuda.to_device(coloring_orders)
    colorings_dev = cuda.to_device(np.zeros_like(coloring_orders))

    # Find colorings
    blocks_per_grid, threads_per_block = 30, 1
    gcc.color_graphs[blocks_per_grid, threads_per_block](strict_graph_dev, coloring_orders_dev, colorings_dev)
    return colorings_dev.copy_to_host()


def random_orderings(strict_graph: np.ndarray, k_orderings: int):
    # Generate orderings
    orderings = gs.generate_random_orderings(strict_graph.shape[0], k_orderings, np.random.default_rng())

    # Transfer data to GPU
    strict_graph_dev = cuda.to_device(strict_graph)
    coloring_orders_dev = cuda.to_device(orderings)
    colorings_dev = cuda.to_device(np.zeros_like(orderings))

    # Find colorings
    blocks_per_grid, threads_per_block = 100, (k_orderings // 100) + 1
    gcc.color_graphs[blocks_per_grid, threads_per_block](strict_graph_dev, coloring_orders_dev, colorings_dev)
    return colorings_dev.copy_to_host()


def backtracking_algorithm(strict_graph: np.ndarray, n_colors: int):
    colorings = NumbaList()
    colorings.append(np.zeros(len(strict_graph), dtype=np.uint16))
    colorings.pop()
    ecb.coloring(
        graph=strict_graph,
        current_coloring=np.zeros(len(strict_graph), dtype=np.uint16),
        vertex=0,
        vertex_color=1,
        n_colors=n_colors,
        colorings_out=colorings
    )
    return np.stack(colorings)


def main(graph_path: str, group_sizes_path: str, n_random: int = 10, k_tables: int = 2):
    title_half_width = 45
    print(f"\nLoading graph from {graph_path}.\n")
    preferences_graph = pd.read_csv(graph_path, index_col=False, header=None).values
    strict_graph = (preferences_graph == 1).astype(int)
    group_sizes = pd.read_csv(group_sizes_path, index_col=False, header=None).values.flatten()  # s_v

    print(f"\n\n{'=' * title_half_width} Naive algorithms {'=' * title_half_width}\n")

    # Stage I: solve problem for hard constraints
    colorings = naive_orderings(strict_graph)
    print(f"Stage I. Find colorings:")
    print_colorings(colorings, labels=["Largest first", "Smallest last"])

    # Stage II
    second_stage(colorings, preferences_graph, strict_graph, group_sizes)

    print(f"\n\n{'=' * title_half_width} Random orderings {'=' * title_half_width}\n")

    colorings = random_orderings(strict_graph, n_random)
    print(f"Stage I. Find colorings:")
    print_colorings(colorings)

    # Stage II
    second_stage(colorings, preferences_graph, strict_graph, group_sizes)

    print(f"\n\n{'=' * (title_half_width - 2)} Backtracking {'=' * (title_half_width - 2)}\n")

    colorings = backtracking_algorithm(strict_graph, k_tables)
    print(f"Stage I. Find colorings:")
    print_colorings(colorings)

    # Stage II
    second_stage(colorings, preferences_graph, strict_graph, group_sizes)


def second_stage(colorings: np.ndarray,
                 preferences_graph: np.ndarray,
                 strict_graph: np.ndarray,
                 group_sizes: np.ndarray):
    # Stage II: xplore the space of feasible solutions
    kempe_chains_dict, kempe_chain_colors_dict = {}, {}
    total_num_of_colorings = len(colorings)
    for parent_coloring_id, coloring in enumerate(colorings):
        kempe_chains, kempe_chain_colors = ks.find_kempe_chains(strict_graph, colorings[parent_coloring_id])
        kempe_chains_dict[parent_coloring_id] = kempe_chains
        total_num_of_colorings += len(kempe_chains)
        kempe_chain_colors_dict[parent_coloring_id] = kempe_chain_colors
    print(f"Stage II. Kempe chains:")
    print_kempe_chains(kempe_chains_dict)

    # Generate new colorings using Kempe chains
    all_colorings = np.zeros((total_num_of_colorings, colorings.shape[1]), dtype=int)
    latest_coloring_id = len(colorings)
    all_colorings[:latest_coloring_id] = colorings
    for (parent_coloring_id, chains), chain_colors in zip(kempe_chains_dict.items(), kempe_chain_colors_dict.values()):
        for chain, colors in zip(chains, chain_colors):
            all_colorings[latest_coloring_id] = colorings[parent_coloring_id].copy()
            ks.swap_kempe_chain_colors(all_colorings[latest_coloring_id], np.array(chain), colors)
            latest_coloring_id += 1
    unique_all_colorings = np.unique(all_colorings, axis=0)
    total_num_of_unique_colorings = len(unique_all_colorings)
    print(f"All colorings (with Kempe chains modification, without duplicates):")
    print_colorings(unique_all_colorings)

    # Find values of cost functions for every coloring
    costs_f1 = np.zeros(total_num_of_unique_colorings)
    costs_f2 = costs_f1.copy()
    for ind, coloring in enumerate(unique_all_colorings):
        costs_f1[ind] = cost_func_1(preferences_graph, group_sizes, coloring)
        costs_f2[ind] = cost_func_2(group_sizes, coloring)
    combined_costs = costs_f1 + costs_f2
    print(f"F1 cost function values: {costs_f1}\n"
          f"F2 cost function values: {costs_f2}\n"
          f"Combined cost function values: {combined_costs}\n"
          f"(smaller is better)\n")

    # Find minimum cost
    best_coloring_id = np.where(combined_costs == np.min(combined_costs))[0]
    if len(best_coloring_id) == 1:
        print(f"The best coloring is {unique_all_colorings[best_coloring_id[0]]} "
              f"with cost {combined_costs[best_coloring_id]}")
    else:
        print(f"The best colorings are: \n{unique_all_colorings[best_coloring_id]},\n "
              f"the cost of each of them is {combined_costs[best_coloring_id][0]}.\n")


if __name__ == '__main__':
    main("../Wedding-seating-plan.csv",
         "../Wedding-seating-plan-Group-sizes.csv",
         k_tables=3)
