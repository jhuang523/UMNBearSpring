# import openkarst
import pykasso as pk
import importlib
importlib.reload(pk)
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import networkx as nx
import openkarst.models
importlib.reload(openkarst.models)
import pickle
from utils.openkarst_network import OpenKarstNetwork as OKN
from openkarst.network_generation import compute_conduit_lengths
from openkarst.visualization.animation_pyvista import animate_network
from openkarst.models import FlowSimulation
from argparse import ArgumentParser
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src"))) #use this to be able to import local packages

def load_network_data(nodes_file, edges_file, diameters_file = None, debug = False, **params):
    """Load network data from csv files and create OpenPNM geometry object"""
    node_keys = params.get('node_keys', {'inlet':['inlet'], 'outlet': ['outfall', 'outlet']})
    network = OKN(nodes_file=nodes_file, edges_file=edges_file, diameters_file=diameters_file)
    cn_geometry = network.load_cave_data(debug = debug, **params)
    inlets, outlets = network.extract_boundary_nodes(node_keys, debug = debug)
    return network
def load_recharge_data(recharge_file):
    """Load recharge data from json file with structure {(node,) : {flow : [], time : []}}"""
    rech_dict = json.load(open(recharge_file))
    return rech_dict

def load_validation_data(validation_file):
    """Load validation data from csv file"""
    val_df = pd.read_csv(validation_file)
    return val_df

def load_results(results_npz):
    results = np.load(results_npz)
    return results

def load_metadata(metadata_file):
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    return metadata
def run_openkarst_simulation(network : OKN, cn_params = None, initial_flowrate = None, initial_water_depth = None, inflow_boundary = {}, head_boundary ={}, steady_state = True, t_max = 1000, inflow_type = 'constant', head_type = 'constant', **params):
    save_path = params.get("save_path")

    adaptive_timesteps = params.get('adaptive_timesteps', True)
    # Setup flow simulation parameters
    physical_properties = {
        'water_density': 1000,        # kg/m^3
        'gravity': 9.81,              # m/s^2
        'dynamic_viscosity': 0.001,   # Pa.s (kg/m.s)
        'geometry_channel': False,    # Channel geometry for analytical solutions (Default False)
        'channel_type': 'infinite',   # 'infinite' for infinitely wide channel, 'finite' for defined width
        'channel_width': 1.0,         # Width of the channel (only used if channel_type is 'finite')
    }
    
    solver_settings = {
        'relaxation_factor': 0.6,    # Dimensionless
        'max_iterations': 20,        # Maximum Picard iterations
        'picard_depth_tol': 1e-5,    # Picard depth tolerance (meters)
        'ss_rel_l2tol': 1e-3,         # L2 tolerance for steady-state
        'ss_rel_madtol': 1e-8         # Median tolerance for steady-state
    }
    
    simulation_settings = {
        'min_waterdepth': 1e-10,      # Minimum water depth (meters)
        'min_flowrate': 1e-10,        # Minimum flow rate (m^3/s)
        'courant': 0.8,               # Courant number
        'adaptive_timesteps': adaptive_timesteps,   # Use adaptive timestepping
        'dt_init': params.get('dt_init', 1.0),             # Initial (or constant) timestep (seconds)
        'dt_max': params.get('dt_max', 1.0),                # Maximum allowable time step
        'steady_state': steady_state,         # Steady-state (True) or transient (False)
        't_max': t_max,              # Maximum time for transient simulations (seconds)
        'print_info_interval': 1000,     # Print info every # time steps
    }
    
    output_settings = {
        'output_interval': 10.0,
        'time': True,
        'time_step_size': True,
        'flowrates': True,
        'water_depths': True,
        'l2_norms': True,
        'convergence_fails': True,
        'reynolds_numbers': True,
    }
    
    logging_settings = {
        'base_dir': save_path if save_path is not None else '.',
        'log_file': 'simulation.log'
    }
    
    # Compute conduit lengths using the utility function
    cn_geometry = compute_conduit_lengths(network.geometry)
    
    # Assign conduit properties
    cn_geometry['throat.epsilon'] = 0.03 # Roughness height (m), default, if cn_params is not none, it will be updated

    if cn_params is not None:
        for param, val in cn_params.items():
            print(param)
            cn_geometry[param] = val

    # Create flow network object
    flow_network = FlowSimulation(cn_geometry,
                                  physical_properties = physical_properties,
                                  solver_settings = solver_settings,
                                  simulation_settings = simulation_settings,
                                  logging_settings = logging_settings)
    
    # Set initial conditions
    if type(initial_flowrate) is float:
        initial_Q = np.full(cn_geometry.Nt, initial_flowrate, dtype=float)   # Initial flows at each conduit (Nt throats)
    elif type(initial_flowrate) is np.ndarray: 
        initial_Q = initial_flowrate #if initial conditions is passed as an array indexed by conduit id
    if type(initial_water_depth) is float:
        initial_y = np.full(cn_geometry.Np, initial_water_depth, dtype=float)  # Initial water depths at each node (Np pores)    
    elif type(initial_water_depth) is np.ndarray:
        initial_y = initial_water_depth #if initial conditions is passed as an array indexed by node id
    flow_network.set_initial_conditions(initial_Q, initial_y)
    
    # Set boundary conditions
    for n in inflow_boundary.keys():
        if inflow_type == 'ramp':
            inflow_values = ('ramp', inflow_boundary[n]['flow_start'], inflow_boundary[n]['time_start'], inflow_boundary[n]['flow_end'], inflow_boundary[n]['time_end'])
        elif inflow_type == 'timeseries':
            inflow_values = ('timeseries', inflow_boundary[n]['time'], inflow_boundary[n]['flow'])
        elif inflow_type == 'constant':
            inflow_values = inflow_boundary[n]['flow']
        flow_network.set_inflow_BC(nodes = list(n), values = inflow_values)

    for n in head_boundary.keys():
        if head_type == 'ramp':
            head_values = ('ramp', head_boundary[n]['head_start'], head_boundary[n]['time_start'], head_boundary[n]['head_end'], head_boundary[n]['time_end'])
        elif head_type == 'timeseries':
            head_values = ('timeseries', head_boundary[n]['time'], head_boundary[n]['head'])
        elif head_type == 'constant':
            head_values = head_boundary[n]['head']
        flow_network.set_waterdepth_BC(nodes = list(n), values = head_values)

    

    # Run simulation and store results
    results = flow_network.run_simulation(desired_outputs = output_settings)
    
    # Get arrays from results container
    Q_history = results['flowrates']
    y_history = results['water_depths']
    t_history = results['time']
    
    
    animation_settings = {
        'update_interval': 1,
        'conduit_plotradius': 0.5,
        'bar_plotradius': 0.5,
        'node_plotsize': 5,
        'depthscaling': 20,
        'fig_width': 1600,
        'fig_height': 800,
        'zoom_factor': 1.0,
        'background_color': 'black',
        'isometric_view': False,
        'create_animation': False,
        'filename': "network_animation2.mp4"
    }
    
    animate_network(cn_geometry=cn_geometry, 
                    Q_history=Q_history, 
                    y_history=y_history, 
                    t_history=t_history, 
                    **animation_settings)
    if save_path is not None:
        try:
            os.makedirs(save_path, exist_ok=False)
        except FileExistsError:
            print("Directory exists. Make sure you don't want to overwrite it.")
            pass
        input_data = {'nodes' : network.nodes, 'edges' : network.edges, 
                      'point_inlets': network.point_inlets,
                      'diffuse_inlets': network.diffuse_inlets,
                      'outlets': network.outlets,
                      'cn_geometry' : cn_geometry, 
                      'inflow_boundary' : inflow_boundary, 
                      'head_boundary' : head_boundary, 
                      'initial_depth' : initial_water_depth, 
                      'initial_flowrate' : initial_flowrate}
        with open(f'{save_path}/input_data.pkl', 'wb') as f:
            pickle.dump(input_data, f)
            print (f'results saved to {save_path}')    

        # Save large numeric arraylike data efficiently (easy to open and extract)
        Q = results['flowrates']
        y = results['water_depths']
        t = results['time']
        re = results['reynolds_numbers']
        np.savez_compressed(f'{save_path}/results_arrays.npz', Q=Q, y=y, t=t, re=re)
    return results

if __name__ == "__main__":
    parser = ArgumentParser(description="Run OpenKarst simulation on a given network")
    parser.add_argument('--network_dir', type=str, required=True, help='Directory containing network files')
    parser.add_argument('--nodes_file', type=str, default='nodes.csv', help='CSV file for nodes')
    parser.add_argument('--edges_file', type=str, default='edges.csv', help='CSV file for edges')
    parser.add_argument('--diameters_file', type=str, default=None, help='CSV file for diameters')
    parser.add_argument('--diffuse_inlets', action='store_true', help='Flag to extract diffuse inlets')
    parser.add_argument('save_path', type=str, help='Path to save simulation results')
    parser.add_argument('recharge_file', type=str, help='CSV file for recharge data')

    args = parser.parse_args()
    network_dir = args.network_dir
    pass
network_dir = '../../data/networks/single_conduit/pykasso_n58'
network = load_network_data(f'{network_dir}/nodes.csv', f'{network_dir}/edges.csv')
network.extract_diffuse_inlets()