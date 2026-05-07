# import openkarst
import sys
import os
import time
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src"))) #use this to be able to import local packages
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../src"))) #use this to be able to import local packages
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src"))) #use this to be able to import local packages
import numpy as np
import pandas as pd
import pickle
import pyarrow.parquet as pq
from utils.openkarst_network import OpenKarstNetwork as OKN
from utils.common import load_yaml, print_verbose, load_pickle, write_pickle
from openkarst.network_generation import compute_conduit_lengths
from openkarst.visualization.animation_pyvista import animate_network
from openkarst.models import FlowSimulation
from argparse import ArgumentParser
from scripts.openkarst.extract_steady_state_conditions import extract_steady_state_conditions
from utils.spring import outlet_flow, write_outlet_flow




def load_network_data(nodes_file, edges_file, diameters_file = None, debug = False, **params):
    """Load network data from csv files and create OpenPNM geometry object"""
    node_keys = params.get('node_keys', {'inlet':['inlet'], 'outlet': ['outfall', 'outlet']})
    network = OKN(nodes_file=nodes_file, edges_file=edges_file, diameters_file=diameters_file)
    cn_geometry = network.load_cave_data(debug = debug, **params)
    inlets, outlets = network.extract_boundary_nodes(node_keys, debug = debug)
    return network

def load_recharge_data(file_path, columns = ['time', 'R_l [V/T]', 'R_h [V/T]']):
    df = pd.read_csv(file_path)
    #check that df has correct columns
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in recharge data.")
    return df

def load_initial_conditions(initial_conditions_file):
    """Load initial conditions from pkl file with structure {'initial_flowrate' : float or array, 'initial_water_depth' : float or array}"""
    init_cond = load_pickle(initial_conditions_file)
    if 'initial_flowrate' not in init_cond.keys():
        init_cond['initial_flowrate'] = 0.0
        print("No initial flowrate specified, setting to 0.0")
    if 'initial_water_depth' not in init_cond.keys():
        init_cond['initial_water_depth'] = 0.0
        print("No initial water depth specified, setting to 0.0")
    return init_cond

def load_metadata(metadata_file):
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    return metadata
def write_inflow_boundary(nodes, Q, t):
    R = {nodes : {'flow' : Q, 'time' : t}}
    return R

def write_partitioned_inflow_boundary(network : OKN, R_l=None, R_h=None, t_l=None, t_h=None):
    R = {}
    if R_l is not None: 
        if t_l is None:
             return ValueError("Time is None")
        n_diffuse = len(network.diffuse_inlets)
        R_l_per_inlet = R_l / n_diffuse 
        R_diffuse = write_inflow_boundary(network.diffuse_inlets, R_l_per_inlet, t_l)
        R.update(R_diffuse)
    if R_h is not None:
        if t_h is None:
            return ValueError("Time is None")
        n_point = len(network.inlets)
        R_h_per_inlet = R_h / n_point 
        R_point = write_inflow_boundary(network.inlets, R_h_per_inlet, t_h)
        R.update(R_point)
    #check R is not empty and all values are real
    if R == {}:
        raise ValueError("No inflow boundary conditions provided.")
    for inlet, data in R.items():
        if not np.isreal(data['flow']).all():
            raise ValueError(f"Recharge values for inlet {inlet} contain non-real numbers.")
    return R

def write_constant_head_boundary(network : OKN, h):
    if network.outlets is None or len(network.outlets) == 0:
        raise ValueError("No outlets found in the network.")
    HB = {network.outlets : {'head' : h}}
    return HB 

def write_input_data(network : OKN, scenario_name: str, R_l=None, R_h=None, t_l=None, t_h=None, h=None, 
                     flow_bound_path = None, 
                     head_bound_path = None, 
                     debug = False):
    if R_l is not None or R_h is not None:
        inflow_boundary= write_inflow_boundary(network, R_l, R_h, t_l, t_h)
    else:
        raise ValueError("At least one of R_l or R_h must be provided.")
    if h is not None:
        head_boundary = write_constant_head_boundary(network, h)
    else:
        raise ValueError("Constant head boundary condition must be provided.")
    #write data to pkl
    if flow_bound_path is not None:
        write_pickle(os.path.join(flow_bound_path, f'{scenario_name}.pkl'), inflow_boundary)
    if head_bound_path is not None:
        write_pickle(os.path.join(head_bound_path, f'{scenario_name}.pkl'), head_boundary)
    print_verbose(f"Input data for scenario {scenario_name} written to {flow_bound_path} and {head_bound_path}.", debug)

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
        'output_interval': params.get('output_interval', 100.0),
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
    
    

    show_animation = params.get('show_animation', False)
    if show_animation:
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
    input_data = {'nodes' : network.nodes, 'edges' : network.edges, 
                    'point_inlets': network.point_inlets,
                    'diffuse_inlets': network.diffuse_inlets,
                    'outlets': network.outlets,
                    'cn_geometry' : cn_geometry, 
                    'inflow_boundary' : inflow_boundary, 
                    'head_boundary' : head_boundary, 
                    'initial_depth' : initial_water_depth, 
                    'initial_flowrate' : initial_flowrate}

    write_pickle(f'{save_path}/input_data.pkl', input_data)
    print (f'results saved to {save_path}')    

    # Save large numeric arraylike data efficiently (easy to open and extract)
    Q = results['flowrates']
    y = results['water_depths']
    t = results['time']
    re = results['reynolds_numbers']
    np.savez_compressed(f'{save_path}/results_arrays.npz', Q=Q, y=y, t=t, re=re)
    return results

def spin_up_simulation(network : OKN, baseflow, head_boundary, head_type, t_ss_max=864000, dt_max = 1000, adaptive_timesteps = True, ss_output_dir=None, init_conditions_file = None, cn_params = None, verbose = False, Q_tol = 1e-3, h_tol = 1e-3):
    initial_flowrate = 1e-5
    initial_water_depth = 1e-5
    inflow_boundary = {network.diffuse_inlets + network.inlets : {'flow' : baseflow / len(network.diffuse_inlets + network.inlets)}}
    ss_results = run_openkarst_simulation(network, 
                        cn_params = cn_params,
                        initial_flowrate = initial_flowrate,
                        initial_water_depth = initial_water_depth,
                        inflow_boundary= inflow_boundary,
                        head_boundary= head_boundary,
                        steady_state = False,
                        dt_max = dt_max,
                        t_max = t_ss_max,
                        adaptive_timesteps = adaptive_timesteps,
                        inflow_type = 'constant',
                        head_type = head_type,
                        save_path = ss_output_dir)
    print_verbose(f"Steady state simulation completed. Extracting steady state conditions...", verbose)
    IC = extract_steady_state_conditions(f'{ss_output_dir}/results_arrays.npz', Q_tol = Q_tol, h_tol = h_tol)
    write_pickle(init_conditions_file, IC)
    return IC

def run_from_yaml(
    input_data_file,
    output_dir=None,
    verbose=False,
):
    input_data = load_yaml(input_data_file)
    print_verbose(f'loaded input data from {input_data_file}', verbose)
    network_dir = input_data.get('network_dir', '.')
    nodes_file = input_data.get('nodes_file', 'nodes.csv')
    edges_file = input_data.get('edges_file', 'edges.csv')
    diameters_file = input_data.get('diameters_file', None)
    output_dir = output_dir if output_dir is not None else input_data.get('output_dir', 'output') 
    inflow_file = input_data.get('inflow_file', 'inflow_boundary.pkl')
    init_conditions_file = input_data.get('initial_conditions_file', 'initial_conditions.pkl')
    head_boundary_file = input_data.get('head_boundary_file', {})
    steady_state = input_data.get('steady_state', False)
    print_verbose(f'steady state set to {steady_state}', verbose)
    cn_params = input_data.get('cn_params', None)
    adaptive_timesteps = input_data.get('adaptive_timesteps', True)
    dt_max = float(input_data.get('dt_max', 1000))
    t_max = float(input_data.get('t_max', 10000))
    head_type = input_data.get('head_boundary_type', 'constant')
    inflow_type = input_data.get('inflow_type', 'constant')
    try: 
        network = load_network_data(f'{network_dir}/{nodes_file}', f'{network_dir}/{edges_file}', diameters_file=diameters_file)
    except FileNotFoundError as e:
        raise Exception(f"Error loading network data: {e}. Skipping")
    inflow_data = load_pickle(inflow_file)
    head_boundary_data = load_pickle(head_boundary_file)
    if init_conditions_file not in [None, 'None']:
        init_conditions = load_initial_conditions(init_conditions_file)
        initial_flowrate = init_conditions['initial_flowrate']
        initial_water_depth = init_conditions['initial_water_depth']
    else:
        initial_flowrate = 0.0
        initial_water_depth = 0.0
    print_verbose(f'inflow: {inflow_data}', verbose)
    print_verbose(f'head boundary: {head_boundary_data}', verbose)
    print_verbose(f'init conditions: Q = {initial_flowrate}, h = {initial_water_depth}', verbose)
    print_verbose(f't_max: {t_max}', verbose)
    run_openkarst_simulation(network, 
                                cn_params = cn_params, 
                                initial_flowrate = initial_flowrate, 
                                initial_water_depth = initial_water_depth, 
                                inflow_boundary= inflow_data, 
                                head_boundary= head_boundary_data, 
                                steady_state = steady_state, 
                                dt_max = dt_max, 
                                t_max = t_max,
                                adaptive_timesteps = adaptive_timesteps,
                                inflow_type = inflow_type, 
                                head_type = head_type, 
                                save_path = output_dir)

def full_simulation_pipeline(input_file, debug = False, **params):
    #load recharge and head params
    input_params = load_yaml(input_file)
    recharge_file = input_params['recharge_file']
    recharge_data = load_recharge_data(recharge_file)
    recharge_distribution = input_params.get('recharge_distribution', 'partitioned')
    head_boundary = input_params['head_boundary']
    head_boundary_file = input_params.get('head_boundary_file', None)
    head_type = input_params.get('head_boundary_type', 'constant')
    inflow_type = input_params.get('inflow_type', 'constant')
    #load network data
    network_dir = input_params['network_dir']
    network_name = ""
    for p in network_dir.split('/'):
        if p not in ['.', 'networks']:
            if network_name == "":
                network_name = p
            else:
                network_name += f'_{p}'
    nodes_file = input_params.get('nodes_file', 'nodes.csv')
    edges_file = input_params.get('edges_file', 'edges.csv')
    network = load_network_data(f'{network_dir}/{nodes_file}', f'{network_dir}/{edges_file}', debug= debug)
    network.extract_diffuse_inlets()
    if not network.network_validity():
        raise Exception("Network validity check failed.")
    #load timestep and other simulation params
    cn_params = input_params.get('cn_params', None)
    adaptive_timesteps = input_params.get('adaptive_timesteps', True)
    dt_max = float(input_params.get('dt_max', 1000))
    t_max = float(input_params.get('t_max', 10000))
    steady_state = input_params.get('steady_state', False)
    spin_up = input_params.get('spin_up', False)
    Q_tol = input_params.get('Q_tol', 1e-3)
    h_tol = input_params.get('h_tol', 1e-3)


    #head boundary 
    if head_boundary_file is not None:
        head_boundary = load_pickle(head_boundary_file)
    else: 
        head_boundary = write_constant_head_boundary(network, head_boundary)
    print_verbose(f'head boundary conditions written', debug)

    #initial conditions 
    init_conditions_file = input_params.get('initial_conditions_file', None)
    if init_conditions_file is not None:
        IC_exists = Path(init_conditions_file).exists()
    baseflow= input_params.get('baseflow', 1e-5)
    ss_output_dir = input_params.get('ss_output_dir', f'output/spinup/{network_name}')
    t_ss_max = input_params.get('t_ss_max', 86400*10)



    if spin_up or not IC_exists: #if manually overriding init conditions or if init conditions file doesn't exist, run spin up simulation to extract steady state conditions
        print_verbose(f"Running spin-up simulation to find steady state conditions for {network_name}", debug)
        IC = spin_up_simulation(network, baseflow, head_boundary, head_type, t_ss_max, dt_max, adaptive_timesteps, ss_output_dir, init_conditions_file, cn_params, debug, Q_tol, h_tol)
        initial_flowrate = IC['initial_flowrate']
        initial_water_depth = IC['initial_water_depth']
    elif IC_exists: #load initial conditions from file
        init_conditions = load_initial_conditions(init_conditions_file)
        initial_flowrate = init_conditions['initial_flowrate']
        initial_water_depth = init_conditions['initial_water_depth']
    else: #set default initial conditions
        initial_flowrate = 1e-5
        initial_water_depth = 1e-5
    print_verbose(f'initial conditions: Q = {initial_flowrate}, h = {initial_water_depth}', debug)
    #write input data 
    R_l = recharge_data['R_l [V/T]']
    R_h = recharge_data['R_h [V/T]']
    t = recharge_data['time']

    for i in range(len(recharge_distribution)):
        r_dist = recharge_distribution[i]
        output_dir = input_params.get('output_dir', f'output/{network_name}/{r_dist}')
        if r_dist == 'partitioned':
            inflow_boundary = write_partitioned_inflow_boundary(network, R_l= R_l, R_h= R_h, t_l = t, t_h = t)
        elif r_dist == 'diffuse':
            inflow_boundary = write_partitioned_inflow_boundary(network, R_l = R_l + R_h, t_l = t, t_h = t)
        elif r_dist == 'point':
            inflow_boundary = write_partitioned_inflow_boundary(network, R_h = R_l + R_h, t_l = t, t_h = t)
        print_verbose(f'inflow boundary conditions written', debug)

        results = run_openkarst_simulation(network, 
                                    cn_params = cn_params, 
                                    initial_flowrate = initial_flowrate, 
                                    initial_water_depth = initial_water_depth, 
                                    inflow_boundary= inflow_boundary, 
                                    head_boundary= head_boundary, 
                                    steady_state = steady_state, 
                                    dt_max = dt_max, 
                                    t_max = t_max,
                                    adaptive_timesteps = adaptive_timesteps,
                                    inflow_type = inflow_type, 
                                    head_type = head_type, 
                                    save_path = output_dir)
        Q = results['flowrates']
        y = results['water_depths']
        t = results['time']
        spring = outlet_flow(network, Q)
        hydrograph_output_path = input_params.get('hydrograph_output_path', 'output/hydrographs/')
        metadata_path = input_params.get('metadata_path', 'output/metadata/')
        run_id = params.get('run_id', int(time.time()))
        run_id = f'{run_id}_{i}'
        #metadata 
        metadata = {
            'run_id' : run_id,
            'network_name' : network_name,
            'recharge_file' : recharge_file,
            'recharge_distribution' : r_dist,
            'baseflow' : baseflow,
            'output_dir' : output_dir,
        }
        metadata_df = pd.DataFrame(metadata, index = [0])
        metadata_df.to_parquet(f'{metadata_path}', partition_cols=['run_id'], compression='snappy')
        print_verbose(f'Metadata written to {metadata_path} with run_id {run_id}', debug)
        write_outlet_flow(t, spring, hydrograph_output_path, run_id=run_id, verbose=debug)
    

    

if __name__ == "__main__":
    parser = ArgumentParser(description="Run OpenKarst simulation on a given network")
    parser.add_argument('--input_data_file', type=str, default=None, help='Path to input data yaml file')
    parser.add_argument('--output_dir', type=str, default = None, help='Path to save simulation results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument("--kwargs",nargs="*",default=[],help="Additional keyword arguments as key=value pairs")

    args = parser.parse_args()
    input_data_file = args.input_data_file
    output_dir = args.output_dir
    verbose = args.verbose
    input_data = load_yaml(input_data_file)
    print_verbose(f'loaded input data from {input_data_file}', verbose)
    kwargs = {} #TODO: fix to process this 
    for item in args.kwargs:
        key, value = item.split("=", 1)
        kwargs[key] = value
    run_from_yaml(input_data_file, output_dir, verbose)
    

    
