"""Works with pykasso v1 for 3D conduit generation"""
# import pykasso as pk
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt 
import plotly.graph_objects as go

from utils.common import print_verbose


# #TODO fix this is still V0 
# def generate_network(settings_file):
#     catchment = pk.SKS(settings_file)
#     catchment.compute_karst_network()
#     network = catchment.karst_simulations[-1]
#     return network

def plot_network(network):
    plt.imshow(network)

def flip_row_index(network_arr):
    """Adjusts indexing so that 0,0 is at top left (for most numpy style ops)"""
    return np.flipud(network_arr)

#fix to use V1 
def generate_n_networks(n_iter, settings_file, output_dir, fname, verbose = False):
    os.makedirs(output_dir, exist_ok = True)
    """Generate n iterations of networks and save bool array as .npy"""
    for i in range(n_iter):
        network = generate_network(settings_file)
        network_array = flip_row_index(network.maps['karst'][0])
        nodes = network.network['nodes']
        nodes_df = pd.DataFrame.from_dict(nodes, orient = 'index').reset_index().rename({'index' : 'id', 0 : 'y', 1 : 'x', 2 : 'type'}, axis = 1)
        edges = network.network['edges']
        edges_df = pd.DataFrame.from_dict(edges, orient = 'index').rename({ 0 : 'from_id', 1 : 'to_id'}, axis = 1)
        uuid = len(os.listdir(output_dir))
        path = f'{output_dir}/{uuid}'
        os.makedirs(path)
        network_path = f'{path}/{fname}.npy'
        node_path = f'{path}/nodes.csv'
        edge_path = f'{path}/edges.csv'
        np.save(network_path, network_array)
        nodes_df.to_csv(node_path, index = False)
        edges_df.to_csv(edge_path, index = False)
        print_verbose(f'{path} saved', verbose)
    print_verbose(f'generated {n_iter} networks', verbose)

def extract_edge_coordinates(nodes, edges):
    """Given a df with nodes with x, y, z coordinates and edges df with to_id and from_id, return appended edges df with x_0, y_0, z_0, x_1, y_1, z_1 columns"""
    edges = edges.merge(
    nodes[['id', 'x', 'y', 'z']],
    left_on='from_id',
    right_on='id',
    how='left'
    ).rename(columns={'x': 'x_0', 'y': 'y_0', 'z': 'z_0'}).drop(columns='id')


    # merge "to" node coordinates
    edges = edges.merge(
    nodes[['id', 'x', 'y', 'z']],
    left_on='to_id',
    right_on='id',
    how='left'
    ).rename(columns={'x': 'x_1', 'y': 'y_1', 'z': 'z_1'}).drop(columns='id')


    return edges

def reset_node_ids(nodes, edges):
    """Resets the node ids such that there are no gaps in the numbering"""
    node_map = {old_id: new_id for new_id, old_id in enumerate(nodes.id)}
    #replace from and to ids in edges
    edges['from_id'] = edges.from_id.map(node_map)
    edges['to_id'] = edges.to_id.map(node_map)
    nodes['id'] = nodes.id.map(node_map)
    return nodes, edges

def densify_edges(nodes, edges, density_factor=3):
    """
    Densify a 3D graph network by adding density_factor evenly spaced
    points along each edge (without changing shape).
    
    nodes: DataFrame with columns ['id','x','y','z']
    edges: DataFrame with columns ['from_id','to_id']
    density_factor: number of interior nodes to insert per edge
    """

    new_nodes = []
    new_edges = []

    next_id = nodes['id'].max() + 1

    for _, row in edges.iterrows():
        # original endpoints
        n1 = nodes.loc[nodes['id'] == row['from_id']].iloc[0]
        n2 = nodes.loc[nodes['id'] == row['to_id']].iloc[0]

        x1, y1, z1 = n1['x'], n1['y'], n1['z']
        x2, y2, z2 = n2['x'], n2['y'], n2['z']

        # Create interpolated nodes
        interp_ids = []
        for k in range(1, density_factor + 1):
            t = k / (density_factor + 1)

            x = (1 - t) * x1 + t * x2
            y = (1 - t) * y1 + t * y2
            z = (1 - t) * z1 + t * z2

            new_nodes.append([next_id, x, y, z])
            interp_ids.append(next_id)
            next_id += 1

        # Subdivide the edge into smaller edges
        pts = [row['from_id']] + interp_ids + [row['to_id']]
        for a, b in zip(pts[:-1], pts[1:]):
            new_edges.append([a, b])

    # 
    new_nodes_df = pd.DataFrame(new_nodes, columns=['id','x','y','z'])
    new_nodes_df['type'] = 'junction'  # add type column
    dense_nodes = pd.concat(
        [nodes, new_nodes_df],
        ignore_index=True
    )

    dense_edges = pd.DataFrame(new_edges, columns=['from_id','to_id'])
    dense_edges = extract_edge_coordinates(dense_nodes, dense_edges)

    return dense_nodes, dense_edges
def reduce_edge_density(nodes, edges, epsilon):
    import networkx as nx
    import rdp

    # Build graph
    G = nx.Graph()
    for _, row in nodes.iterrows():
        G.add_node(row['id'],
                   pos=np.array([row['x'], row['y'], row['z']]),
                   type=row['type'])

    for _, row in edges.iterrows():
        G.add_edge(row['from_id'], row['to_id'])

    # Extract polylines
    def get_polylines(G):
        polylines = []
        visited = set()

        for node in G.nodes():
            if G.degree(node) != 2:  # endpoints and junctions
                for nbr in G.neighbors(node):

                    if (node, nbr) in visited:
                        continue

                    path = [node, nbr]
                    visited.add((node, nbr))

                    prev, curr = node, nbr

                    while G.degree(curr) == 2:
                        nxt = [n for n in G.neighbors(curr) if n != prev][0]

                        if (curr, nxt) in visited:
                            break

                        path.append(nxt)
                        visited.add((curr, nxt))
                        prev, curr = curr, nxt

                    polylines.append(path)

        return polylines

    polylines = get_polylines(G)

    # Simplify polylines while preserving endpoints
    simplified_polylines = []

    for path in polylines:

        coords = np.array([G.nodes[n]['pos'] for n in path])

        simplified_coords = rdp.rdp(coords, epsilon=epsilon)

        simplified_polylines.append(simplified_coords)

    # Rebuild graph WITHOUT duplicating shared nodes
    coord_to_id = {}
    new_nodes = []
    new_edges = []
    node_id = 0

    def get_or_create_node(coord):

        nonlocal node_id

        key = tuple(np.round(coord, 8))

        if key not in coord_to_id:

            coord_to_id[key] = node_id

            new_nodes.append([
                node_id,
                coord[0],
                coord[1],
                coord[2],
                'junction'
            ])

            node_id += 1

        return coord_to_id[key]

    for poly in simplified_polylines:

        prev_id = None

        for coord in poly:

            curr_id = get_or_create_node(coord)

            if prev_id is not None:
                new_edges.append([prev_id, curr_id])

            prev_id = curr_id

    nodes_df = pd.DataFrame(
        new_nodes,
        columns=['id', 'x', 'y', 'z', 'type']
    )

    edges_df = pd.DataFrame(
        new_edges,
        columns=['from_id', 'to_id']
    )

    # Restore inlet/outlet types using nearest match
    original_special = nodes[nodes['type'] != 'junction']

    for _, row in original_special.iterrows():

        coord = np.array([row['x'], row['y'], row['z']])

        dists = np.linalg.norm(
            nodes_df[['x','y','z']].values - coord,
            axis=1
        )

        idx = np.argmin(dists)

        nodes_df.loc[idx, 'type'] = row['type']
    edges_df = extract_edge_coordinates(nodes_df, edges_df)
    return nodes_df, edges_df

def conduit_lengths(nodes, edges): 
    #extract edge coordinates if not already existing
    for coord in ['x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']:
        if coord not in edges.columns:
            edges = extract_edge_coordinates(nodes, edges)
            break
    edges['length'] = ((edges.x_0 - edges.x_1)**2 + (edges.y_0 - edges.y_1)**2 + (edges.z_0 - edges.z_1)**2)**0.5
    return edges 
            
def gaussian_diameter_distribution(nodes, edges, d):
    return
def plot_3D_network(
    nodes,
    edges,
    node_color=None,
    edge_color=None,
    node_colormap='viridis',
    edge_colormap='viridis'
):
    # --------------------
    # EDGES
    # --------------------
    line_traces = []
    lines = edges[["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]].values.reshape(-1, 2, 3)

    if isinstance(edge_color, str) and edge_color in edges.columns:
        # color edges by column
        for (_, row), line in zip(edges.iterrows(), lines):
            line_traces.append(
                go.Scatter3d(
                    x=line[:, 0],
                    y=line[:, 1],
                    z=line[:, 2],
                    mode='lines',
                    line=dict(
                        color=row[edge_color],
                        colorscale=edge_colormap,
                        width=1
                    ),
                    showlegend=False
                )
            )
    else:
        # single color (default or explicit)
        color = edge_color if edge_color is not None else 'blue'
        for line in lines:
            line_traces.append(
                go.Scatter3d(
                    x=line[:, 0],
                    y=line[:, 1],
                    z=line[:, 2],
                    mode='lines',
                    line=dict(color=color, width=1),
                    showlegend=False
                )
            )

    # --------------------
    # NODES
    # --------------------
    if isinstance(node_color, str) and node_color in nodes.columns:
        # color nodes by column
        if nodes[node_color].dtype == 'O':
            node_vals = nodes[node_color].astype('category').cat.codes
        else:
            node_vals = nodes[node_color]
        showscale = True

    else:
        # single color (default or explicit)
        node_vals = node_color if node_color is not None else 'red'
        showscale = False

    scatter = go.Scatter3d(
        x=nodes.x,
        y=nodes.y,
        z=nodes.z,
        mode='markers',
        marker=dict(
            size=5,
            color=node_vals,
            colorscale=node_colormap,
            showscale=showscale
        ),
        showlegend=False
    )

    # --------------------
    # FIGURE
    # --------------------
    fig = go.Figure(data=line_traces + [scatter])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig.show()

