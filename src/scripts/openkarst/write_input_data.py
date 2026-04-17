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
def write_inflow_boundary(network : okn.OpenKarstNetwork, R_l=None, R_h=None, t_l=None, t_h=None):
    R = {}
    if R_l is not None: 
        if t_l is None:
             return ValueError("Time is None")
        n_diffuse = len(network.diffuse_inlets)
        R_l_per_inlet = R_l / n_diffuse 
        R[network.diffuse_inlets] = {'flow' : R_l_per_inlet, 'time' : t_l}
    if R_h is not None:
        if t_h is None:
            return ValueError("Time is None")
        n_point = len(network.inlets)
        R_h_per_inlet = R_h / n_point 
        R[network.inlets] = {'flow' : R_h_per_inlet, 'time' : t_h}
    #check R values are real
    for inlet, data in R.items():
        if not np.isreal(data['flow']).all():
            raise ValueError(f"Recharge values for inlet {inlet} contain non-real numbers.")
    return R

def write_constant_head_boundary(network : okn.OpenKarstNetwork, h):
    if network.outlets is None or len(network.outlets) == 0:
        raise ValueError("No outlets found in the network.")
    HB = {network.outlets : {'head' : h}}
    return HB 

