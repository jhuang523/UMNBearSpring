import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src"))) #use this to be able to import local packages
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../src"))) #use this to be able to import local packages
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src"))) #use this to be able to import local packages
import utils.geos
import utils.openkarst_network as okn
import utils.conduits
from utils.common import write_pickle, print_verbose


def write_inflow_boundary(nodes, Q, t):
    R = {nodes : {'flow' : Q, 'time' : t}}
    return R

def write_partitioned_inflow_boundary(network : okn.OpenKarstNetwork, R_l=None, R_h=None, t_l=None, t_h=None):
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

def write_constant_head_boundary(network : okn.OpenKarstNetwork, h):
    if network.outlets is None or len(network.outlets) == 0:
        raise ValueError("No outlets found in the network.")
    HB = {network.outlets : {'head' : h}}
    return HB 


def write_input_data(network : okn.OpenKarstNetwork, scenario_name: str, R_l=None, R_h=None, t_l=None, t_h=None, h=None, flow_bound_path = 'input_data/inflow_boundary', head_bound_path = 'input_data/head_boundary', print_verbose = False):
    if R_l is not None or R_h is not None:
        inflow_boundary= write_inflow_boundary(network, R_l, R_h, t_l, t_h)
    else:
        raise ValueError("At least one of R_l or R_h must be provided.")
    if h is not None:
        head_boundary = write_constant_head_boundary(network, h)
    else:
        raise ValueError("Constant head boundary condition must be provided.")
    #write data to pkl
    write_pickle(os.path.join(flow_bound_path, f'{scenario_name}.pkl'), inflow_boundary)
    write_pickle(os.path.join(head_bound_path, f'{scenario_name}.pkl'), head_boundary)
    print_verbose(f"Input data for scenario {scenario_name} written to {flow_bound_path} and {head_bound_path}.", print_verbose)

