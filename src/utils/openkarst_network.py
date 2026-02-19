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
        self.geometry = None
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
  
    def load_cave_data(self,debug = False, **params):
        """
        Loads the cave data from the CSV files and constructs a NetworkX graph.

        This method reads the node coordinates, edge connections, and node diameters
        from their respective CSV files. It constructs a NetworkX graph with the
        loaded data, where nodes have coordinates and edges have average diameters
        based on the connected nodes. The two diameters available at each node are
        currently averaged. The graph is then converted into an OpenPNM geometry
        object with assigned conduit lengths and diameters.

        Returns:
            openpnm.network.GenericNetwork: An OpenPNM geometry object representing
                the network with assigned conduit lengths and diameters.
        """
        
        G = nx.Graph()
        node_diameters = {}
        #if node data is provided, use it directly
        if self.nodes is not None: 
            nodes = self.nodes
            if 'z' not in nodes.columns:
                nodes['z'] = params.get('z', 0)
            if 'd' not in nodes.columns:
                nodes['d']= params.get('d', 1) #assign uniform value
            nodes = nodes.set_index('id')

            coords_array = nodes[['x', 'y', 'z']].to_numpy()  # shape (N, 3)
            node_ids = nodes.index.to_numpy()

            nodes_list = list(zip(node_ids, [{'coords': coord.tolist()} for coord in coords_array]))
            G.add_nodes_from(nodes_list)
            G = nx.relabel_nodes(G, lambda x: int(x))
            if debug:
                print(G.nodes)
        # Load nodes and their coordinates from the file, skipping the header
        else:
            if self.nodes_file.endswith('.csv'):
                nodes = pd.read_csv(self.nodes_file)
                if 'z' not in nodes.columns:
                    nodes['z'] = params.get('z', 0)
                if 'd' not in nodes.columns:
                    nodes['d']= params.get('d', 1) #assign uniform value

                coords_array = nodes[['x', 'y', 'z']].to_numpy()  # shape (N, 3)
                node_ids = nodes.id.to_numpy()

                nodes_list = list(zip(node_ids, [{'coords': coord.tolist()} for coord in coords_array]))
                G.add_nodes_from(nodes_list)
                G = nx.relabel_nodes(G, lambda x: int(x))
                if debug:
                    print(G.nodes)
            else:
                #TODO: fix so this stores .txt files as dfs 
                with open(self.nodes_file, 'r') as file:
                    next(file)  # Skip the header line
                    for line in file:
                        node_id, x, y, z = line.strip().split(';')
                        G.add_node(int(node_id), coords=[float(x), float(y), float(z)])

        #load edges
        if self.edges is not None:
            edges = self.edges
            edge_data = edges[['from_id', 'to_id']].astype(int).itertuples(index=False, name=None)
            G.add_edges_from(edge_data)
            if debug:
                print(edge_data)
        else:
            if self.edges_file.endswith('.csv'):
                edges = pd.read_csv(self.edges_file)
                edge_data = edges[['from_id', 'to_id']].astype(int).itertuples(index=False, name=None)
                G.add_edges_from(edge_data)
                if debug:
                    print(edge_data)
            else:
                #TODO: fix so this stores .txt files as dfs 
                with open(self.edges_file, 'r') as file:
                    next(file)  # Skip the header line
                    for line in file:
                        node_a, node_b = map(int, line.strip().split(';'))
                        G.add_edge(node_a, node_b)
        
        #check node numbering is regular 
        self.update_network(nodes = nodes, edges=edges)
        
        # Load diameters from the file, skipping the header
        if self.diameters is not None:
            node_diameters = self.diameters.set_index('id').to_dict()['d']
            nodes['d'] = nodes['id'].map(node_diameters)
        elif self.diameters_file is not None: #to do fix this so it can take in CSV
            node_diameters = {}
            with open(self.diameters_file, 'r') as file:
                next(file)  # Skip the header line
                for line in file:
                    node_id, cswidth, csheight = line.strip().split(';')
                    average_diameter = (float(cswidth) + float(csheight)) / 2
                    node_diameters[int(node_id)] = average_diameter
            nodes['d'] = nodes['id'].map(node_diameters)
        else:
            node_diameters = nodes.set_index('id')['d'].to_dict()
        self.update_network(nodes=nodes, diameters = node_diameters)       
        
        # Load edges from the file, skipping the header

        
        ## check edges, nodes, diameters 
        # Assign average diameters to each edge by averaging diameters of connected nodes
        edge_diameters = {}
        for node_a, node_b in G.edges():
            avg_diameter = (node_diameters[node_a] + node_diameters[node_b]) / 2
            edge_diameters[tuple(sorted((node_a, node_b)))] = avg_diameter
            
        # Create an openPNM geometry object
        cn_geometry = op.io.network_from_networkx(G)
        
        # Compute and assign conduit lengths 
        coords_diff = np.diff(cn_geometry.coords[cn_geometry.conns], axis=1).squeeze()
        squared_diffs = coords_diff**2
        sum_squared_diffs = np.sum(squared_diffs, axis=1)
        conduit_lengths = np.sqrt(sum_squared_diffs)
        cn_geometry['throat.lengths'] = conduit_lengths
        
        # # Assign conduit diameters to openPNM geometry object
        cn_geometry['throat.diameters'] = [edge_diameters[tuple(sorted(edge))] for edge in cn_geometry['throat.conns']]
        self.geometry = cn_geometry
        return cn_geometry
    
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
    
    def store_inlets(self,node_dict):
        """Store inlet nodes from a dictionary that contains 'point_inlets' and 'diffuse_inlets' keys."""
        self.point_inlets += tuple(node_dict['point_inlets'])
        self.diffuse_inlets += tuple(node_dict['diffuse_inlets'])
        print ("Point inlets:", self.point_inlets)
        print ("Diffuse inlets:", self.diffuse_inlets)

    def plot_network_flow(self, Q, h,  **params): #Q and h are arrays that match the number of edges (Q) and number of nodes (h)
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.collections import LineCollection

        axes = params.get("axes", ("x", "y"))
        segments = self.edges[[f'{axes[0]}_0', f'{axes[1]}_0', f'{axes[0]}_1', f'{axes[1]}_1']].values.reshape(-1, 2, 2)
        norm = plt.Normalize(vmin=Q.min(), vmax=Q.max())

        # Get palette
        palette = params.get("palette", "coolwarm")
        cmap = sns.color_palette(palette, as_cmap=True)

        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(Q)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.add_collection(lc)
        ax.autoscale()  # Needed for LineCollection to be visible
        sns.scatterplot(self.nodes, x = axes[0], y = axes[1], hue = h, ax = ax, palette = 'viridis')
        plt.colorbar(lc, ax=ax, label="Flowrate (m³/s)")
        plt.grid(True)
        plt.legend(title = "Head")

    
    def plot_3D_network(self, **params):
        node_color = params.get('node_color', None)
        edge_color = params.get('edge_color', None)
        node_colormap = params.get('node_colormap', 'viridis')
        edge_colormap = params.get('edge_colormap', 'viridis')
        conduits.plot_3D_network(self.nodes, self.edges, 
                        node_color=node_color, 
                        edge_color=edge_color,
                        node_colormap=node_colormap,
                        edge_colormap=edge_colormap)