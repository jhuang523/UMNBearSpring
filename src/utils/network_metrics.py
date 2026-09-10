import pandas as pd
import numpy as np
import os
import sys
import networkx as nx
from utils.conduits import *
def reduce_graph(nodes, edges, preserve_types=('inlet', 'outlet')):
    """
    Reduce a conduit network by collapsing degree-2 intermediate nodes.

    Parameters
    ----------
    nodes : pd.DataFrame
        Must contain:
            id
            type

    edges : pd.DataFrame
        Must contain:
            from_id
            to_id

        Any additional edge properties are preserved/aggregated
        where appropriate.

    preserve_types : tuple
        Node types that should always be retained.

    Returns
    -------
    nodes_reduced : pd.DataFrame
    edges_reduced : pd.DataFrame
    G_reduced : nx.Graph
    """

    # ---------------------------------------------------------
    # Build graph from nodes and edges
    # ---------------------------------------------------------

    G = nx.Graph()

    for _, row in nodes.iterrows():
        G.add_node(
            row['id'],
            **row.to_dict()
        )

    for _, row in edges.iterrows():
        G.add_edge(
            row['from_id'],
            row['to_id'],
            **row.to_dict()
        )

    # ---------------------------------------------------------
    # Nodes to preserve
    # ---------------------------------------------------------

    keep = set(
        nodes.loc[
            nodes['type'].isin(preserve_types),
            'id'
        ]
    )

    # Always preserve junctions/endpoints
    keep.update(
        n for n in G.nodes
        if G.degree[n] != 2
    )

    # ---------------------------------------------------------
    # Reduce graph
    # ---------------------------------------------------------

    G_reduced = nx.Graph()

    # Add preserved nodes
    for node in keep:
        G_reduced.add_node(
            node,
            **G.nodes[node]
        )

    visited = set()

    for start in keep:

        for neighbor in G.neighbors(start):

            edge_key = frozenset((start, neighbor))

            if edge_key in visited:
                continue

            path_nodes = [start]
            path_edges = []

            previous = start
            current = neighbor

            while True:

                edge_key = frozenset((previous, current))

                if edge_key in visited:
                    break

                visited.add(edge_key)

                path_edges.append(
                    G.edges[previous, current]
                )

                path_nodes.append(current)

                # Stop when reaching another important node
                if current in keep:
                    break

                # Continue through degree-2 node
                next_nodes = [
                    n for n in G.neighbors(current)
                    if n != previous
                ]

                if len(next_nodes) != 1:
                    break

                previous = current
                current = next_nodes[0]

            # -------------------------------------------------
            # Add collapsed edge
            # -------------------------------------------------

            if current in keep and current != start:

                edge_data = {}

                # Total length
                if 'length' in edges.columns:
                    edge_data['length'] = sum(
                        e['length']
                        for e in path_edges
                    )

                # Length-weighted diameter
                if 'diameter' in edges.columns:

                    if 'length' in edges.columns:
                        weights = np.array([
                            e['length']
                            for e in path_edges
                        ])
                    else:
                        weights = np.ones(len(path_edges))

                    diameters = np.array([
                        e['diameter']
                        for e in path_edges
                    ])

                    edge_data['diameter'] = np.average(
                        diameters,
                        weights=weights
                    )

                G_reduced.add_edge(
                    start,
                    current,
                    **edge_data
                )

    # ---------------------------------------------------------
    # Convert back to DataFrames
    # ---------------------------------------------------------

    nodes_reduced = nodes[
        nodes['id'].isin(G_reduced.nodes)
    ].copy()

    edges_reduced = pd.DataFrame([
        {
            'from_id': u,
            'to_id': v,
            **data
        }
        for u, v, data in G_reduced.edges(data=True)
    ])

    return nodes_reduced, edges_reduced, G_reduced

def fully_reduce_graph(nodes, edges, preserve_types=('inlet', 'outlet'), check_node_type = 'junction'):
    fully_reduced = False 
    reduced_nodes = nodes.copy()
    reduced_edges = edges.copy()
    while not fully_reduced:
        reduced_nodes, reduced_edges, G_reduced = reduce_graph(reduced_nodes, reduced_edges, preserve_types=preserve_types)
        # reduced_edges = extract_edge_coordinates(reduced_nodes, reduced_edges)
        reduced_nodes['degree'] = reduced_nodes.id.map(dict(G_reduced.degree()))
        nodes_to_check = reduced_nodes[(reduced_nodes.degree == 2) & (reduced_nodes.type.isin([check_node_type]))]
        if nodes_to_check.empty:
            fully_reduced = True
    return reduced_nodes, reduced_edges, G_reduced

def bounding_box(nodes):
    x_min = nodes.x.min()
    x_max = nodes.x.max()
    y_min = nodes.y.min()
    y_max = nodes.y.max()
    z_min = nodes.z.min()
    z_max = nodes.z.max()
    return x_min, x_max, y_min, y_max, z_min, z_max
def bounding_box_2D(nodes):
    x_min = nodes.x.min()
    x_max = nodes.x.max()
    y_min = nodes.y.min()
    y_max = nodes.y.max()
    return x_min, x_max, y_min, y_max
def index(nodes, edges, mode, extent_method = 'bounding_box',verbose = True):
    temp_nodes = nodes.copy()
    temp_edges = edges.copy()
    if mode == 'HI':
        temp_nodes['z'] = 0 # set all nodes to the same elevation to create a 2D network
        temp_edges['z_0'] = 0
        temp_edges['z_1'] = 0
    temp_edges = conduit_lengths(temp_nodes, temp_edges)
    total_length = temp_edges.length.sum()
    print_verbose(f"Total conduit length: {total_length}", verbose)
    if extent_method == 'bounding_box':
        x_min, x_max, y_min, y_max, z_min, z_max = bounding_box(temp_nodes)
        if mode == 'HI':
            max_extent = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        elif mode == "VI":
            max_extent =(z_max - z_min)
        else:
            max_extent = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2 + (z_max - z_min)**2)

        print_verbose(f"Max extent, bounding box: {max_extent}", verbose)
    elif extent_method == 'straight_line':
        max_extent = 0
        for outlet in nodes[nodes.type == 'outlet'].itertuples():
            for inlet in nodes[nodes.type == 'inlet'].itertuples():
                d = np.sqrt((outlet.x - inlet.x)**2 + (outlet.y - inlet.y)**2 + (outlet.z - inlet.z)**2)
                max_extent += d
        print_verbose(f"Max extent, straight line distance: {max_extent}", verbose)           
    #TODO: other methods
    index = total_length/max_extent
    return index

def VI(nodes, edges, extent_method = 'bounding_box', verbose = True):
    temp_nodes = nodes.copy()
    temp_edges = edges.copy()
    temp_edges = conduit_lengths(temp_nodes, temp_edges)
    total_length = temp_edges.length.sum()
    print_verbose(f"Total conduit length: {total_length}", verbose)
    if extent_method == 'bounding_box':
        x_min, x_max, y_min, y_max, z_min, z_max = bounding_box(temp_nodes)
        max_extent = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2 + (z_max - z_min)**2)
        print_verbose(f"Max extent of network: {max_extent}", verbose)
    VI = total_length/max_extent
    return VI

def tortuosity(nodes, edges, mode = "sum"):
    reduced_nodes, reduced_edges, G_reduced = fully_reduce_graph(nodes, edges)
    reduced_edges = extract_edge_coordinates(reduced_nodes, reduced_edges)
    reduced_edges = conduit_lengths(reduced_nodes, reduced_edges)
    full_edges = extract_edge_coordinates(nodes, edges)
    full_edges = conduit_lengths(nodes, full_edges)
    if mode == "sum":
        tortuosity = full_edges.length.sum()/reduced_edges.length.sum()
    #TODO: add other methods for calculating tortuosity
    return tortuosity

def orientation_entropy(nodes, edges, bins = 36):
    edges = extract_edge_coordinates(nodes, edges)
    edges = conduit_lengths(nodes, edges)
    # Calculate the orientation of each edge in degrees
    orientations = np.degrees(np.arctan2(edges.y_1 - edges.y_0, edges.x_1 - edges.x_0))
    # Create a histogram of orientations
    hist, bin_edges = np.histogram(orientations, bins=bins, range=(-180, 180), density=False)
    hist = hist/hist.sum()  # Normalize the histogram to get a probability distribution
    # Calculate the entropy of the orientation distribution
    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log(hist_nonzero)/np.log(bins))
    return orientations, hist, entropy

def alpha_index(nodes, edges):
    n, e = len(nodes), len(edges)
    return 2 * (e - n + 1) / ((n -1) * (n - 2))

def beta_index(nodes, edges):
    n, e = len(nodes[nodes.type.isin(['inlet', 'outlet'])]), len(edges)
    return e / n

