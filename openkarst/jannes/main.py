import pandas as pd 
import numpy as np
from openkarst.network_generation import compute_conduit_lengths
from openkarst.models import FlowSimulation
import networkx as nx
import openpnm as op


def main(debug = True):
    #load network data
    network_dir = 'anastomotic_simple_clean'
    nodes_file = 'nodes.csv'
    edges_file = 'edges.csv'
    nodes = pd.read_csv(f'{network_dir}/{nodes_file}')
    edges = pd.read_csv(f'{network_dir}/{edges_file}')
    inlets = tuple(nodes[nodes['type'] == 'inlet']['id'].tolist())
    outlets = tuple(nodes[nodes['type'] == 'outlet']['id'].tolist())
    diffuse_inlets = tuple(nodes[nodes['type'] == 'junction']['id'].tolist())
    G = nx.Graph()
    node_diameters = {}
    #if node data is provided, use it directly

    nodes = nodes.set_index('id')

    coords_array = nodes[['x', 'y', 'z']].to_numpy()  # shape (N, 3)
    node_ids = nodes.index.to_numpy()

    nodes_list = list(zip(node_ids, [{'coords': coord.tolist()} for coord in coords_array]))
    G.add_nodes_from(nodes_list)
    G = nx.relabel_nodes(G, lambda x: int(x))
    #load edges

    edge_data = edges[['from_id', 'to_id']].astype(int).itertuples(index=False, name=None)
    G.add_edges_from(edge_data)
    
    try:
        node_diameters = nodes.set_index('id')['d'].to_dict()
    except KeyError:
        node_diameters = nodes['d'].to_dict()
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

    #boundary conditions
    inflow_type = 'constant'
    baseflow= 0.02
    initial_Q = 1e-5
    initial_y = 1e-5
    inflow_boundary = {diffuse_inlets + inlets : {'flow' : baseflow / len(diffuse_inlets + inlets)}}

    head_boundary = {outlets : {'head' : 0.01}}
    head_type = 'constant'

    ss_output_dir =  f'output/{network_dir}'
    t_ss_max = 86400*20
    #running manually and checking for convergence 
    # Run simulation and store results

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
        'adaptive_timesteps': True,   # Use adaptive timestepping
        'dt_init': 1,             # Initial (or constant) timestep (seconds)
        'dt_max': 1e3,                # Maximum allowable time step
        'steady_state': False,         # Steady-state (True) or transient (False)
        't_max': t_ss_max,              # Maximum time for transient simulations (seconds)
        'print_info_interval': 1000,     # Print info every # time steps
    }
    
    output_settings = {
        'output_interval': 100.0,
        'time': True,
        'time_step_size': True,
        'flowrates': True,
        'water_depths': True,
        'l2_norms': True,
        'convergence_fails': True,
        'reynolds_numbers': True,
    }
    
    logging_settings = {
        'base_dir': ss_output_dir,
        'log_file': 'simulation.log'
    }

    
    # Compute conduit lengths using the utility function
    cn_geometry = compute_conduit_lengths(cn_geometry)
    
    # Assign conduit properties
    cn_geometry['throat.epsilon'] = 0.03 # Roughness height (m), default, if cn_params is not none, it will be updated
    # Create flow network object
    flow_network = FlowSimulation(cn_geometry,
                                  physical_properties = physical_properties,
                                  solver_settings = solver_settings,
                                  simulation_settings = simulation_settings,
                                  logging_settings = logging_settings)
    
    # Set initial conditions

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

    #write input data to pkl for record keeping

    results = flow_network.run_simulation(desired_outputs = output_settings)

    # Save large numeric arraylike data efficiently (easy to open and extract)
    Q = results['flowrates']
    y = results['water_depths']
    t = results['time']
    re = results['reynolds_numbers']
    np.savez_compressed(f'{ss_output_dir}/results_arrays.npz', Q=Q, y=y, t=t, re=re)
    return results

    
if __name__ == "__main__":
    main()