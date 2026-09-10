import numpy as np
import pandas as pd 
import networkx as nx
import openpnm as op
from openkarst.models import FlowSimulation

def load_network(nodes, edges, **params):
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
        
    edge_data = edges[['from_id', 'to_id']].astype(int).itertuples(index=False, name=None)
    G.add_edges_from(edge_data)

    node_diameters = dict(zip(node_ids, nodes['d']))
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
    return cn_geometry

def run_reservoir_simulation(reservoirs, inflow_boundary, save_path):
    #TODO: update save path
    adaptive_timesteps = True
    steady_state = False 
    output_interval = 100
    dt_init = 1.0
    dt_max = 100
    inflow_type = 'timeseries'
    head_type = 'constant'


    head_boundary = {outlets: {'head' : 0.001}}
    t_max= recharge_time[-1]


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
        "relaxation_factor": 0.6,
        "max_iterations": 20,
        "picard_depth_tol": 1e-6,
        "ss_rel_l2tol": 1e-5,
    }

    simulation_settings = {
        'min_waterdepth': 1e-10,      # Minimum water depth (meters)
        'min_flowrate': 1e-10,        # Minimum flow rate (m^3/s)
        'courant': 0.8,               # Courant number
        'adaptive_timesteps': adaptive_timesteps,   # Use adaptive timestepping
        'dt_init': dt_init,             # Initial (or constant) timestep (seconds)
        'dt_max': dt_max,                # Maximum allowable time step
        'steady_state': steady_state,         # Steady-state (True) or transient (False)
        't_max': t_max,              # Maximum time for transient simulations (seconds)
        'print_info_interval': 10000,     # Print info every # time steps
    }

    output_settings = {
        'output_interval': output_interval,  # Output results every # time steps
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
    cn_geometry = network

    # Assign conduit properties
    cn_geometry['throat.epsilon'] = 0.03 # Roughness height (m), default, if cn_params is not none, it will be updated

    # Set initial conditions-- no flow
    initial_Q = np.full(cn_geometry.Nt, 0, dtype=float)   # Initial flows at each conduit (Nt throats)
    initial_y = np.full(cn_geometry.Np, 1e-3, dtype=float)   # Initial depths at each conduit (Np pores)

    # Create flow network object
    flow_network = FlowSimulation(cn_geometry,
                                    physical_properties = physical_properties,
                                    solver_settings = solver_settings,
                                    simulation_settings = simulation_settings,
                                    logging_settings = logging_settings)


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

    #!!! Add reservoirs 
    for reservoir_node, reservoir in reservoirs.items():
        flow_network.add_reservoir(
                reservoir_node,
                reservoir['A'],
                reservoir['S_y'],
                reservoir['h_0'],
                reservoir['C'],
                recharge = reservoir['reservoir_recharge'],
            )

    #write input data to pkl for record keeping
    input_data = {'nodes' : nodes, 'edges' : edges, 
                'point_inlets': inlets,
                'diffuse_inlets': diffuse_inlets,
                'outlets': outlets,
                'cn_geometry' : cn_geometry, 
                'inflow_boundary' : inflow_boundary, 
                'head_boundary' : head_boundary, 
                'initial_depth' : initial_y, 
                'initial_flowrate' : initial_Q,
                'reservoirs' : reservoirs}

    #observation points at springs    
    flow_network.set_observation_points(
        nodes=observation_nodes,
        variables=["connected_abs_flowrate", "connected_net_flowrate"],
        interval=output_settings["output_interval"],
    )
    flow_network.set_observation_points(
        nodes=list(reservoirs.keys()),
        variables=["reservoir_head", "reservoir_storage", "reservoir_exchange"],
        interval=output_settings["output_interval"]
    )
    # Run simulation and store results

    results = flow_network.run_simulation(desired_outputs = output_settings)

    # Get observation data as DataFrame

    obs_df = flow_network.get_observation_dataframe()
    saved_results = {}

    obs_df.to_csv(f"{save_path}/observation_data.csv", index=False)

    np.savez(f"{save_path}/results.npz", **results)

# Network loading
nodes = pd.read_csv('nodes.csv')
edges = pd.read_csv('edges.csv')
network = load_network(nodes, edges)
inlets = (0,)
outlets = (1,)
diffuse_inlets = [node for node in nodes.id if node not in inlets and node not in outlets]
observation_nodes = outlets

#Recharge Loading
recharge_data = pd.read_csv('gaussian_pulse.csv')
recharge_timeseries = np.array(recharge_data.recharge)
recharge_time = np.array(recharge_data.time)
conduit_eta = 0.5
matrix_eta = 1 - conduit_eta

#Uniform Reservoirs
save_path = "uniform_reservoirs"
reservoirs = {
    node : {'A' : 1e3/len(diffuse_inlets), 'S_y' : 0.15, 'h_0' : .001, 'C' : 1e-4, 'reservoir_recharge' : ('timeseries', recharge_time, recharge_timeseries*matrix_eta/len(diffuse_inlets))} for node in diffuse_inlets

}

inflow_boundary = {inlets : {'flow' : recharge_timeseries * conduit_eta, 'time': recharge_time}}

run_reservoir_simulation(reservoirs, inflow_boundary, save_path)

#Uniform Reservoirs
save_path_high_C = "uniform_high_C"
reservoirs_high_C = {
    node : {'A' : 1e3/len(diffuse_inlets), 'S_y' : 0.15, 'h_0' : .001, 'C' : 1e-2, 'reservoir_recharge' : ('timeseries', recharge_time, recharge_timeseries*matrix_eta/len(diffuse_inlets))} for node in diffuse_inlets

}

inflow_boundary = {inlets : {'flow' : recharge_timeseries * conduit_eta, 'time': recharge_time}}

run_reservoir_simulation(reservoirs_high_C, inflow_boundary, save_path_high_C)

#2 reservoirs with different conductance
save_path_2_res = "two_reservoirs"
reservoirs_2_res = {
    10 : {'A' : 1e3/2, 'S_y' : 0.15, 'h_0' : 0.001, 'C' : 1e-4, 'reservoir_recharge' : ('timeseries', recharge_time, recharge_timeseries*matrix_eta/2)},
    15: {'A' : 1e3/2, 'S_y' : 0.15, 'h_0' : 0.001, 'C' : 1e-2, 'reservoir_recharge' : ('timeseries', recharge_time, recharge_timeseries*matrix_eta/2)},
}

inflow_boundary = {inlets : {'flow' : recharge_timeseries * conduit_eta, 'time': recharge_time}}

run_reservoir_simulation(reservoirs_2_res, inflow_boundary, save_path_2_res)