#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:22:54 2024

@author: Jannes Kordilla
@contact: jannes.kordilla@idaea.csic.es

Modified by Jenny Huang (huan1428@umn.edu) to be more general and support pykasso 
"""

import networkx as nx
import openpnm as op
import numpy as np
import pandas as pd
import utils.conduits as conduits
from utils.common import print_verbose

class OpenKarstNetwork:
    """
    A class to load cave network data from CSV files and create an OpenPNM geometry object.

    This class reads node coordinates, edge connections, and diameters from
    respective CSV files and constructs a NetworkX graph with the data. It then
    converts the graph into an OpenPNM geometry object with assigned conduit lengths
    and diameters.

    Attributes:
        nodes_file (str): Path to the CSV file containing node coordinates.
        edges_file (str): Path to the CSV file containing edge connections.
        diameters_file (str): Path to the CSV file containing node diameters.

    Methods:
        load_cave_data(): Loads the cave data from the CSV files and constructs
            a NetworkX graph with node coordinates, edge connections, and
            edge diameters. Returns an OpenPNM geometry object.
    """
    
    def __init__(self, **data):
        """
        Initializes the CaveDataLoader with file paths.

        Args:
            nodes_file (str): Path to the CSV file containing node coordinates.
            edges_file (str): Path to the CSV file containing edge connections.
            diameters_file (str): Path to the CSV file containing node diameters.
            nodes (pd.DataFrame): DataFrame containing node data.
            edges (pd.DataFrame): DataFrame containing edge data.
            diameters (pd.DataFrame): DataFrame containing diameter data.
        """
        self.nodes_file = data.get('nodes_file')
        self.edges_file = data.get('edges_file')
        self.diameters_file = data.get('diameters_file')
        self.nodes = data.get('nodes')
        self.edges = data.get('edges')
        self.diameters = data.get('diameters')
        self.geometry = data.get('geometry')
        self.graph = None
        self.inlets = ()
        self.diffuse_inlets = ()
        self.point_inlets = ()
        self.outlets = ()
    def update_network(self, **data):
        """Update network data with new dataframes"""
        for key, value in data.items():
            try:
                setattr(self, key, value)
            except Exception as e:
                print(f"Failed to set attribute {key}: {e}")
    def save_data(self, save_path, **params):
        nodes_file = params.get('nodes_file', 'nodes.csv')
        edges_file = params.get('edges_file', 'edges.csv')

        """Save network data to a specified path in CSV format."""
        if self.nodes is not None:
            self.nodes.to_csv(f"{save_path}/nodes.csv", index=False)
        if self.edges is not None:
            self.edges.to_csv(f"{save_path}/edges.csv", index=False)
        print_verbose(f"Data saved to {save_path}", params.get('debug', False))
    def load_cave_data(self, debug=False, **params):
        """Load cave network and return an OpenPNM network."""

        # ------------------------------------------------------------------
        # Nodes
        # ------------------------------------------------------------------
        if self.nodes is not None:
            nodes = self.nodes.copy()
        elif self.nodes_file.endswith(".csv"):
            nodes = pd.read_csv(self.nodes_file)
        else:
            nodes = pd.read_csv(
                self.nodes_file,
                sep=";",
                names=["id", "x", "y", "z"],
                skiprows=1,
            )

        nodes["id"] = nodes["id"].astype(int)

        if "z" not in nodes:
            nodes["z"] = params.get("z", 0)

        if "d" not in nodes:
            nodes["d"] = params.get("d", 1.0)

        # ------------------------------------------------------------------
        # Edges
        # ------------------------------------------------------------------
        if self.edges is not None:
            edges = self.edges.copy()
        elif self.edges_file.endswith(".csv"):
            edges = pd.read_csv(self.edges_file)
        else:
            edges = pd.read_csv(
                self.edges_file,
                sep=";",
                names=["from_id", "to_id"],
                skiprows=1,
            )

        edges[["from_id", "to_id"]] = edges[["from_id", "to_id"]].astype(int)
        edges = conduits.remove_duplicate_edges(edges)

        # ------------------------------------------------------------------
        # Diameters
        # ------------------------------------------------------------------
        if isinstance(self.diameters, (int, float)):
            nodes["d"] = float(self.diameters)

        elif isinstance(self.diameters, pd.DataFrame):
            diameter_map = self.diameters.set_index("id")["d"]
            nodes["d"] = nodes["id"].map(diameter_map)

        elif self.diameters_file is not None:
            d = pd.read_csv(
                self.diameters_file,
                sep=";",
                names=["id", "width", "height"],
                skiprows=1,
            )
            d["d"] = (d["width"] + d["height"]) / 2
            nodes["d"] = nodes["id"].map(d.set_index("id")["d"])

        node_diameters = nodes.set_index("id")["d"].to_dict()

        # ------------------------------------------------------------------
        # NetworkX graph
        # ------------------------------------------------------------------
        G = nx.Graph()

        G.add_nodes_from(
            (
                row.id,
                {"coords": [row.x, row.y, row.z]},
            )
            for row in nodes.itertuples(index=False)
        )

        G.add_edges_from(
            edges[["from_id", "to_id"]].itertuples(index=False, name=None)
        )

        if debug:
            print(G.nodes(data=True))
            print(G.edges())

        self.update_network(
            nodes=nodes,
            edges=edges,
            diameters=node_diameters,
            graph=G,
        )

        # ------------------------------------------------------------------
        # Edge diameters
        # ------------------------------------------------------------------
        edge_diameters = {
            tuple(sorted((u, v))):
            (node_diameters[u] + node_diameters[v]) / 2
            for u, v in G.edges
        }

        # ------------------------------------------------------------------
        # OpenPNM network
        # ------------------------------------------------------------------
        net = op.io.network_from_networkx(G)

        coords = net.coords[net.conns]
        net["throat.lengths"] = np.linalg.norm(coords[:, 0] - coords[:, 1], axis=1)

        net["throat.diameters"] = [
            edge_diameters[tuple(sorted(edge))]
            for edge in net["throat.conns"]
        ]

        self.update_network(geometry=net)

        return net
    
    def extract_edge_coordinates(self, debug = False):
        edges = conduits.extract_edge_coordinates(self.nodes, self.edges)
        self.update_network(edges = edges)
        print_verbose("edges updated", debug)
    
    def degree(self, debug = False):
        if self.graph is None:
            self.load_cave_data(debug = debug)

        degree_df = pd.DataFrame(self.graph.degree(), columns = ['id', 'degree'])
        degree_df = degree_df.set_index('id') 
        self.nodes['degree'] = self.nodes.id.map(degree_df['degree'])
        self.update_network(degree = degree_df)
        return self.nodes
        print_verbose("node degrees calculated", debug)
    def conduit_lengths(self, debug = False): 
        edges = conduits.conduit_lengths(self.nodes, self.edges)
        self.update_network(edges = edges)
        print_verbose("conduit lengths calculated", debug)

    def extract_boundary_nodes(self, node_keys = {'inlet':['inlet'], 'outlet': ['outfall', 'outlet']}, debug = False):
        if self.nodes is not None: 
            nodes = self.nodes
        elif self.nodes_file.endswith('.csv'):
            nodes = pd.read_csv(self.nodes_file)
        inlets = nodes.id[nodes.type.isin(node_keys['inlet'])].to_list()
        outlets = nodes.id[nodes.type.isin(node_keys['outlet'])].to_list()
        self.update_network(inlets = tuple(inlets), outlets = tuple(outlets))
        return inlets, outlets
    
    def extract_diffuse_inlets(self, inlet_label = 'inlet', outlet_label = 'outlet'):
        diffuse_inlets = self.nodes.id[(self.nodes.type != inlet_label) & (self.nodes.type != outlet_label)].to_list()
        self.update_network(diffuse_inlets = tuple(diffuse_inlets))
    
    def extract_lattice_boundary_nodes(self, lattice_no_flow_label = 'lattice_edge', lattice_inlet_label = 'lattice_top'):
        lattice_inlets = self.nodes.id[self.nodes.type == lattice_no_flow_label].to_list()
        lattice_outlets = self.nodes.id[self.nodes.type == lattice_inlet_label].to_list()
        self.update_network(lattice_inlets = tuple(lattice_inlets), lattice_edges = tuple(lattice_outlets))

    def store_inlets(self,node_dict):
        """Store inlet nodes from a dictionary that contains 'point_inlets' and 'diffuse_inlets' keys."""
        self.point_inlets += tuple(node_dict['point_inlets'])
        self.diffuse_inlets += tuple(node_dict['diffuse_inlets'])
        print ("Point inlets:", self.point_inlets)
        print ("Diffuse inlets:", self.diffuse_inlets)

    def update_diameters(self, d): 
        edge_diameters = {}
        #assign constant value
        if isinstance(d, float) or isinstance(d, int): 
            diameters = dict.fromkeys(self.diameters.keys(), d)
        elif isinstance(d, pd.DataFrame):
            diameters = d.set_index('id')['d'].to_dict()
        elif isinstance(d, dict):
            diameters = d
        self.update_network(diameters = diameters)
        self.nodes['d'] = self.nodes.id.map(diameters)
        for node_a, node_b in self.graph.edges():
            avg_diameter = (self.diameters[node_a] + self.diameters[node_b]) / 2
            edge_diameters[tuple(sorted((node_a, node_b)))] = avg_diameter
        self.geometry['throat.diameters'] = [edge_diameters[tuple(sorted(edge))] for edge in self.geometry['throat.conns']]
        return self.geometry

    def calculate_network_volume(self):
        V = (self.geometry['throat.lengths'] * (self.geometry['throat.diameters']/2)**2 * np.pi).sum()
        self.update_network(volume = V)
        return V
    def calculate_water_volume(self, h):
        return 
    def network_validity(self):
        """Check if network has at least 1 inlet, at least 1 outlet, and all nodes are connected to at least one edge."""
        if len(self.inlets) == 0:
            print("Network has no inlets.")
            return False
        if len(self.outlets) == 0:
            print("Network has no outlets.")
            return False
        for node in self.graph.nodes():
            if self.graph.degree(node) == 0:
                print(f"Node {node} is not connected to any edges.")
                return False
        print("Network is valid.")
        return True
    def plot_3D_network(self, **params):
        node_color = params.get('node_color', None)
        edge_color = params.get('edge_color', None)
        node_colormap = params.get('node_colormap', 'viridis')
        edge_colormap = params.get('edge_colormap', 'viridis')
        show_nodes = params.get('show_nodes', True)
        return conduits.plot_3D_network(self.nodes, self.edges, 
                                        show_nodes=show_nodes,
                        node_color=node_color, 
                        edge_color=edge_color,
                        node_colormap=node_colormap,
                        edge_colormap=edge_colormap)
    def plot_network_flow(self, Q, h, **params):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.collections import LineCollection

        axes = params.get("axes", ("x", "y"))
        ax = params.get("ax", None)
        lc = params.get("lc", None)
        sc = params.get("sc", None)
        hide_nodes = params.get("hide_nodes", False)
        edge_lookup = {
            tuple(sorted((row.from_id, row.to_id))): i
            for i, row in self.edges.iterrows()
        }

        order = [
            edge_lookup[tuple(sorted(conn))]
            for conn in self.geometry['throat.conns']
        ]

        edges_ordered = self.edges.iloc[order].reset_index(drop=True)
        segments = edges_ordered[
            [f'{axes[0]}_0', f'{axes[1]}_0',
            f'{axes[0]}_1', f'{axes[1]}_1']
        ].values.reshape(-1, 2, 2)

        palette = params.get("palette", "coolwarm")
        cmap = sns.color_palette(palette, as_cmap=True)

        norm = plt.Normalize(vmin=params.get("vmin_q", Q.min()), vmax=params.get("vmax_q", Q.max()))
        norm_h = plt.Normalize(vmin=params.get("vmin_h", h.min()), vmax=params.get("vmax_h", h.max()))

        # ---------- INIT MODE ----------
        if lc is None:
            if ax is None:
                fig, ax = plt.subplots()
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(Q)
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
            if not hide_nodes: 
                sns.scatterplot(self.nodes, x = axes[0], y = axes[1], hue = h, ax = ax, palette = 'viridis')
                plt.legend(title = "Head")
            cbar = plt.colorbar(lc, ax=ax, label="Flowrate (m³/s)")
            plt.grid(True)
            

            return ax, lc, sc, cbar
        # ---------- UPDATE MODE ----------
        lc.set_array(Q)
        sc.set_array(h)
        return ax, lc, sc, cbar
    
    
    def animate_network_flow(self, Q, h, t, Q_out, **params): #Q and h are arrays that match the number of edges (Q) and number of nodes (h)
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.collections import LineCollection
        from matplotlib.animation import FuncAnimation
        save_path = params.get('save_path', None)

        #get timesteps
        dt = params.get('dt', 3600)
        t0 = t[0]
        frame_indices = []

        target_time = t0
        i = 0

        while i < len(t):
            if t[i] >= target_time:
                frame_indices.append(i)
                target_time += dt
            i += 1

        #INITIALIZE Figure    
        fig, ax = plt.subplots(2, 1, figsize=(6,10))
        ax[0].set_aspect('equal')
        ax[0].set_axis_off()

        # global limits (important!)
        params = dict(
            ax=ax[0],
            vmin_q=Q.min(),
            vmax_q=Q.max(),
            vmin_h=h.min(),
            vmax_h=h.max(),
        )

        # initialize using your function
        lc, sc = self.plot_network_flow(
            Q[0],
            h[0],
            **params
        )
        line, = ax[1].plot(t, Q_out, lw=2, label="Outlet flowrate")
        dot, = ax[1].plot([], [], 'ro', markersize=8)
        spring_ylims = params.get("spring_ylims", (Q_out.min() - Q_out.mean() * 0.1, Q_out.max() + Q_out.mean() * 0.1))
        ax[1].set_xlim(t[0], t[-1])
        ax[1].set_ylim(spring_ylims[0], spring_ylims[1])
        ax[1].legend()
        ax[1].grid(True)
        def update(idx):

            self.plot_network_flow(
                Q[idx],
                h[idx],
                ax=ax[0],
                lc=lc,
                sc=sc,
                params = params
            )

            ax[0].set_title(f"t = {t[idx]}")
            dot.set_data([t[idx]], [Q_out[idx]])

            return lc, sc, dot
        ani = FuncAnimation(
            fig,
            update,
            frames=frame_indices,
            interval=200,
            blit=False
        )
        if save_path is not None:
            ani.save(save_path, writer="pillow", fps=15)
        plt.show()

    # def plot_network_flow(self, Q, h,  **params): #Q and h are arrays that match the number of edges (Q) and number of nodes (h)
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #     from matplotlib.collections import LineCollection

    #     axes = params.get("axes", ("x", "y"))
    #     segments = self.edges[[f'{axes[0]}_0', f'{axes[1]}_0', f'{axes[0]}_1', f'{axes[1]}_1']].values.reshape(-1, 2, 2)
    #     norm = plt.Normalize(vmin=Q.min(), vmax=Q.max())

    #     # Get palette
    #     palette = params.get("palette", "coolwarm")
    #     cmap = sns.color_palette(palette, as_cmap=True)

    #     lc = LineCollection(segments, cmap=cmap, norm=norm)
    #     lc.set_array(Q)

    #     fig, ax = plt.subplots(figsize=(6, 6))
    #     ax.add_collection(lc)
    #     ax.autoscale()  # Needed for LineCollection to be visible
    #     sns.scatterplot(self.nodes, x = axes[0], y = axes[1], hue = h, ax = ax, palette = 'viridis')
    #     plt.colorbar(lc, ax=ax, label="Flowrate (m³/s)")
    #     plt.grid(True)
    #     plt.legend(title = "Head")