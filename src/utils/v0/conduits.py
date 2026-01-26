"""To generate networks with this module, you need to be working with the dev version of pykasso (0.1) and python <=3.9. 
Still trying to troubleshoot HFM (random walk implementation) for higher versions of python."""
import pykasso as pk
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt 
from utils.common import print_verbose



def generate_network(settings_file):
    catchment = pk.SKS(settings_file)
    catchment.compute_karst_network()
    network = catchment.karst_simulations[-1]
    return network

def plot_network(network):
    plt.imshow(network)

def flip_row_index(network_arr):
    """Adjusts indexing so that 0,0 is at top left (for most numpy style ops)"""
    return np.flipud(network_arr)

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
    
    

