"""To generate networks with this module, you need to be working with the dev version of pykasso and python <=3.9. 
Still trying to troubleshoot HFM (random walk implementation) for higher versions of python."""
import pykasso as pk
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt 
from utils.utils import print_verbose



def generate_network(settings_file):
    catchment = pk.SKS(settings_file)
    catchment.compute_karst_network()
    network = catchment.karst_simulations[-1].maps['karst'][0]
    return network

def plot_network(network):
    plt.imshow(network)

def flip_row_index(network):
    """Adjusts indexing so that 0,0 is at top left (for most numpy style ops)"""
    return np.flipud(network)

def generate_n_networks(n_iter, settings_file, output_dir, fname, verbose = False):
    os.makedirs(output_dir, exist_ok = True)
    """Generate n iterations of networks and save bool array as .npy"""
    for i in range(n_iter):
        network = generate_network(settings_file)
        uuid = len(os.listdir(output_dir))
        path = f'{output_dir}/{fname}_{uuid}.npy'
        np.save(path, network)
        print_verbose(f'{path} saved', verbose)
    print_verbose(f'generated {n_iter} networks', verbose)
    
    

