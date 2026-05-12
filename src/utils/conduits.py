"""Works with pykasso v1 for 3D conduit generation"""
# import pykasso as pk
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import utils.geos as geos
from utils.common import print_verbose


# #TODO fix this is still V0 
# def generate_network(settings_file):

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

def save_network(nodes, edges, network_dir, network_name):
    os.makedirs(f'{network_dir}/{network_name}', exist_ok = True)
    nodes.to_csv(f'{network_dir}/{network_name}/nodes.csv', index = False)
    edges.to_csv(f'{network_dir}/{network_name}/edges.csv', index = False)
    print_verbose(f'Network saved to {network_dir}', True)
def extract_edge_coordinates(nodes, edges):
    """Given a df with nodes with x, y, z coordinates and edges df with to_id and from_id, return appended edges df with x_0, y_0, z_0, x_1, y_1, z_1 columns"""
    if 'id' not in nodes.columns: #id is index
        nodes.reset_index(inplace=True)
    edges = edges[['from_id', 'to_id']].merge(
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

def remove_duplicate_edges(edges, from_col = 'from_id', to_col = 'to_id'):
    edges[['id_1', 'id_2']] = np.sort(
        edges[[from_col, to_col]],
        axis=1
    )
    return edges.drop_duplicates(subset=['id_1', 'id_2']).reset_index(drop=True).drop(columns = ['id_1', 'id_2'])
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
def reduce_node_density(nodes, edges, epsilon, protected_types = ['inlet', 'outlet']):
    import numpy as np
    import pandas as pd
    import networkx as nx
    import rdp

    # --- Build graph ---
    G = nx.Graph()

    for _, row in nodes.iterrows():
        G.add_node(
            row['id'],
            pos=np.array([row['x'], row['y'], row['z']]),
            type=row['type']
        )

    for _, row in edges.iterrows():
        G.add_edge(row['from_id'], row['to_id'])

    # --- Find polylines (same as before, but cleaner tracking) ---
    def get_polylines(G):
        polylines = []
        visited_edges = set()

        for node in G.nodes():
            if G.degree(node) != 2:  # endpoints/junctions
                for nbr in G.neighbors(node):

                    edge = tuple(sorted((node, nbr)))
                    if edge in visited_edges:
                        continue

                    path = [node, nbr]
                    visited_edges.add(edge)

                    prev, curr = node, nbr

                    while G.degree(curr) == 2:
                        nxt = [n for n in G.neighbors(curr) if n != prev][0]
                        edge = tuple(sorted((curr, nxt)))

                        if edge in visited_edges:
                            break

                        path.append(nxt)
                        visited_edges.add(edge)

                        prev, curr = curr, nxt

                    polylines.append(path)

        return polylines

    polylines = get_polylines(G)

    # --- Decide which nodes to keep ---
    nodes_to_keep = set()

    for path in polylines:
        coords = np.array([G.nodes[n]['pos'] for n in path])

        mask = rdp.rdp(coords, epsilon=epsilon, return_mask=True)

        # force exact endpoints
        mask[0] = True
        mask[-1] = True


        # map simplified coords back to original node IDs
        for i, keep in enumerate(mask):
            if G.nodes[path[i]]['type'] in protected_types:
                mask[i] = True
                nodes_to_keep.add(path[i])
            elif keep:
                nodes_to_keep.add(path[i])
            
    # --- Build reduced graph ---
    H = nx.Graph()

    for n in nodes_to_keep:
        H.add_node(n, **G.nodes[n])

    # reconnect edges by walking original graph
    for u in nodes_to_keep:
        for v in G.neighbors(u):
            if v not in nodes_to_keep:
                # walk until next kept node
                prev, curr = u, v

                while curr not in nodes_to_keep:
                    nxt = [n for n in G.neighbors(curr) if n != prev][0]
                    prev, curr = curr, nxt

                H.add_edge(u, curr)

            else:
                H.add_edge(u, v)

    # --- Convert back to DataFrames ---
    nodes_out = []
    for n, data in H.nodes(data=True):
        x, y, z = data['pos']
        nodes_out.append([n, x, y, z, data['type']])

    edges_out = []
    for u, v in H.edges():
        edges_out.append([u, v])

    nodes_df = pd.DataFrame(
        nodes_out,
        columns=['id', 'x', 'y', 'z', 'type']
    )

    edges_df = pd.DataFrame(
        edges_out,
        columns=['from_id', 'to_id']
    )
    edges_df = edges_df.drop_duplicates(subset=['from_id', 'to_id']).reset_index(drop=True) # remove any duplicate edges that may have been created
    return nodes_df, edges_df

def adjust_node_spacing(nodes, edges, spacing):
    """
    Add intermediate nodes along edges so that
    node spacing is approximately <= spacing.
    """

    # node lookup
    coord = nodes.set_index("id")[["x", "y", "z"]].to_dict("index")

    new_nodes = []
    new_edges = []

    # start new IDs after existing max
    next_id = nodes["id"].max() + 1
    if ['x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1'] not in edges.columns:
        edges = extract_edge_coordinates(nodes, edges)
    if 'length' not in edges.columns:
        edges = conduit_lengths(nodes, edges)
    for _, edge in edges.iterrows():

        n1 = edge["from_id"]
        n2 = edge["to_id"]
        length = edge['length']
        if length <= spacing:
            new_edges.append({
                "from": n1,
                "to": n2
            })
            continue
        # number of segments needed
        nseg = max(1, int(np.ceil(length / spacing)))

        # interpolation positions
        tvals = np.linspace(0, 1, nseg + 1)

        # create ordered node list along edge
        edge_nodes = [n1]

        # intermediate nodes
        for t in tvals[1:-1]:

            xn = x1 + t * dx
            yn = y1 + t * dy

            new_nodes.append({
                "id": next_id,
                "x": xn,
                "y": yn
            })

            edge_nodes.append(next_id)
            next_id += 1

        edge_nodes.append(n2)

        # connect sequentially
        for a, b in zip(edge_nodes[:-1], edge_nodes[1:]):
            new_edges.append({
                "from": a,
                "to": b
            })

    # combine original + inserted nodes
    all_nodes = pd.concat(
        [nodes, pd.DataFrame(new_nodes)],
        ignore_index=True
    )

    all_edges = pd.DataFrame(new_edges)

    return all_nodes, all_edges
def conduit_lengths(nodes, edges): 
    #extract edge coordinates if not already existing
    for coord in ['x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']:
        if coord not in edges.columns:
            edges = extract_edge_coordinates(nodes, edges)
            break
    edges['length'] = ((edges.x_0 - edges.x_1)**2 + (edges.y_0 - edges.y_1)**2 + (edges.z_0 - edges.z_1)**2)**0.5
    return edges 

def project_conduit_elevations(nodes, edges, grid, x0, y1, dx, dy):
    nodes['z'] = geos.get_elev_from_coords(nodes.x, nodes.y, grid, x0, y1, dx, dy)
    edges = extract_edge_coordinates(nodes, edges)
    return nodes, edges

def conduit_slopes(nodes, edges):
    for coord in ['x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']:
        if coord not in edges.columns:
            edges = extract_edge_coordinates(nodes, edges)
            break
    if 'length' not in edges.columns:
        edges = conduit_lengths(nodes, edges)
    dz = edges.z_1 - edges.z_0
    dx = edges.x_1 - edges.x_0
    dy = edges.y_1 - edges.y_0
    edges['dz_dr'] = dz/(dx**2 + dy**2)**0.5
    return edges

def rescale_slope(nodes, edges, alpha, reference_id):
    nodes = nodes.copy()
    edges = edges.copy()
    z_ref = nodes.loc[nodes.id == reference_id, 'z'].iloc[0]
    nodes['z'] = z_ref + alpha * (nodes['z'] - z_ref)
    edges = extract_edge_coordinates(nodes, edges)
    return nodes, edges


def gaussian_diameter_distribution(nodes, edges, d):
    return

def plot_2D_network(nodes, 
                    edges, 
                    node_color=None, 
                    edge_color=None, 
                    node_colormap='viridis', 
                    edge_colormap='viridis', **params):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.collections import LineCollection

    axes = params.get("axes", ("x", "y"))
    ax = params.get("ax", None)
    lc = params.get("lc", None)
    sc = params.get("sc", None)
    grid = params.get("grid", True)

    segments = edges[
        [f'{axes[0]}_0', f'{axes[1]}_0',
        f'{axes[0]}_1', f'{axes[1]}_1']
    ].values.reshape(-1, 2, 2)

    palette = params.get("palette", "coolwarm")
    cmap = sns.color_palette(palette, as_cmap=True)
    if node_color is not None: 
        if node_color in nodes.columns:
            node_color = nodes[node_color]
        norm_node = plt.Normalize(vmin=params.get("vmin_node", node_color.min()), vmax=params.get("vmax_node", node_color.max()))
    if edge_color is not None:
        if edge_color in edges.columns:
            edge_color = edges[edge_color]
        norm_edge = plt.Normalize(vmin=params.get("vmin_edge", edge_color.min()), vmax=params.get("vmax_edge", edge_color.max()))

    # ---------- INIT MODE ----------
    if lc is None:
        if ax is None:
            fig, ax = plt.subplots()
        if edge_color is not None: 
            lc = LineCollection(segments, cmap=cmap, norm=norm_edge)
            lc.set_array(edge_color)

        else:
            lc = LineCollection(segments, color = 'black')
        ax.add_collection(lc)
        ax.autoscale()

        # sc = ax.scatter(
        #     self.nodes[axes[0]],
        #     self.nodes[axes[1]],
        #     c=h,
        #     cmap="viridis",
        #     norm=norm_h,
        #     s=30
        # )
        if node_color is not None:
            sns.scatterplot(nodes, x = axes[0], y = axes[1], hue = node_color, ax = ax, palette = 'viridis')
        plt.colorbar(lc, ax=ax)
        plt.grid(grid)

        return ax

def plot_3D_network(
    nodes,
    edges,
    show_nodes=True,
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
    if show_nodes:
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
        scatter_trace = [scatter]
    else: 
        scatter_trace = []

    # --------------------
    # FIGURE
    # --------------------
    fig = go.Figure(data=line_traces + scatter_trace)
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    return fig

