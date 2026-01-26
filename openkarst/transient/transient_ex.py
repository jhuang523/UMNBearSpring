# import openkarst
import numpy as np
import pandas as pd
import cave_data_loader as cdl
from openkarst.network_generation import compute_conduit_lengths
from openkarst.models import FlowSimulation

"""files needed
transient_ex.py : main simulation script
cave_loader.py : load cave data from csv files, adapted to allow passing dataframes directly and input conduit information
nodes.csv : node coordinates
edges.csv : edge connections
recharge.npy : recharge data"""
nodes = pd.read_csv(f'nodes.csv')
edges = pd.read_csv(f'edges.csv')

loader = cdl.CaveDataLoader(nodes = nodes, edges = edges)
geometry = loader.load_cave_data(d = 1)
inlets = [18]
outlets = [0]
Q_in = np.load('recharge.npy')  # m3/s

#main simulation function
def main(geometry, base_dir = '.', inlets = [], outlets = [], cn_params = None, flowrate = 0, water_depth = 0.01, inflow_boundary = {}, head_boundary ={}, steady_state = True, t_max = 1000, dt = 1, inflow_type = 'constant', head_type = 'constant', end_time = 3600, **params):
    
    # base_dir = os.path.dirname(os.path.abspath(__file__))
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
        'picard_depth_tol': 1e-7,    # Picard depth tolerance (meters)
        'ss_rel_l2tol': 1e-3,         # L2 tolerance for steady-state
        'ss_rel_madtol': 1e-8         # Median tolerance for steady-state
    }
    
    simulation_settings = {
        'min_waterdepth': 1e-10,      # Minimum water depth (meters)
        'min_flowrate': 1e-10,        # Minimum flow rate (m^3/s)
        'courant': 0.8,               # Courant number
        'adaptive_timesteps': adaptive_timesteps,   # Use adaptive timestepping
        'dt_init': dt,             # Initial (or constant) timestep (seconds)
        'dt_max': 1.0,                # Maximum allowable time step
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
        'base_dir': base_dir,
        'log_file': 'simulation.log'
    }
    
    # Create network object using OpenPNM
    dl = 1 # Constant spacing between nodes (meters)
    cn_geometry = geometry
    
    # Compute conduit lengths using the utility function
    cn_geometry = compute_conduit_lengths(cn_geometry)
    
    # Assign conduit properties
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
    initial_Q = np.full(cn_geometry.Nt, flowrate, dtype=float)   # Initial flows at each conduit (Nt throats)
    initial_y = np.full(cn_geometry.Np, water_depth, dtype=float)  # Initial water depths at each node (Np pores)    
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

    
    # flow_network.set_boundary_conditions(
    # inflow_boundary=inflow_boundary,
    # waterdepth_boundary=waterdepth_boundary,
    # inflow_type=inflow_type,
    # end_time= end_time)

    # Run simulation and store results
    results = flow_network.run_simulation(desired_outputs = output_settings)
    
    # Get arrays from results container
    # Q_history = results['flowrates']
    # y_history = results['water_depths']
    # t_history = results['time']
    
    
    # animation_settings = {
    #     'update_interval': 1,
    #     'conduit_plotradius': 0.5,
    #     'bar_plotradius': 0.5,
    #     'node_plotsize': 5,
    #     'depthscaling': 20,
    #     'fig_width': 1600,
    #     'fig_height': 800,
    #     'zoom_factor': 1.0,
    #     'background_color': 'black',
    #     'isometric_view': False,
    #     'create_animation': False,
    #     'filename': "network_animation2.mp4"
    # }
    
    # animate_network(cn_geometry=cn_geometry, 
    #                 Q_history=Q_history, 
    #                 y_history=y_history, 
    #                 t_history=t_history, 
    #                 **animation_settings)
    return results

results = main(geometry, 
               inlets = inlets, outlets = outlets, 
               cn_params = {'throat.epsilon' : 0.03}, 
               inflow_boundary= {tuple(inlets) : {'flow' : Q_in, 'time' : list(range(len(Q_in)))}}, 
               head_boundary= {tuple(outlets) : {'head' : 0.01}}, 
               steady_state = False, water_depth=0.1, t_max = 86400*30, 
               dt = 1, inflow_type = 'timeseries', adaptive_timesteps = False)