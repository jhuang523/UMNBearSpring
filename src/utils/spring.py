# import openkarst
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src"))) #use this to be able to import local packages
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../src"))) #use this to be able to import local packages

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src"))) #use this to be able to import local packages
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uuid
import time
from utils.common import write_pickle, print_verbose
from argparse import ArgumentParser
from utils.openkarst_network import OpenKarstNetwork

def single_node_flow(Q,node_id, network : OpenKarstNetwork):
    #Q is an nt x ne array of flowrates, node_id is the node we want to calculate flow for
    #returns an nt array of flowrates at the node
    flow = np.zeros(Q.shape[0])
    edges = network.edges
    from_edges = edges[(edges.from_id == node_id)].index
    to_edges = edges[(edges.to_id == node_id)].index
    for i in from_edges:
        flow -= Q[:, i]
    for i in to_edges:
        flow += Q[:, i]
    return flow
def outlet_flow(network : OpenKarstNetwork, Q): 
    edges = network.edges
    outlet_Q = {}
    outlet_ids = network.outlets
    for node in outlet_ids:
    #flow is stored in the Q array as (ts, edge_index)
        outlet_Q[node] = single_node_flow(Q, node, network)
    return outlet_Q

def write_outlet_flow(t, spring_Q, output_path, run_id = None, method = "parquet", partition = ["run_id"], verbose = False):
    df = pd.DataFrame(spring_Q)
    df['time'] = t
    df = df.melt(
        id_vars="time",
        var_name="spring",
        value_name="Q"
    )
    df['run_id'] = run_id if run_id is not None else float(time.time())
    if method == "parquet":
        df.to_parquet(output_path, partition_cols=partition, compression='snappy')
        print_verbose(f"Outlet flow written to {output_path} with run_id {run_id}", verbose)
    elif method == 'dataframe':
        os.makedirs(output_path, exist_ok=True)
        df.to_csv(f'{output_path}/{run_id}.csv', index=False)
    else:
        #TODO: Add other output methods
        raise ValueError("Unsupported output method. Use 'parquet' or 'dataframe'.")

def calculate_recession_coefficients(t, Q_spring, n_coeffs = 3):
    import pwlf
    #get max-- assume recession starts at max flow

    max_idx = np.argmax(Q_spring)
    recession_Q = Q_spring[max_idx:]
    recession_t = t[max_idx:]
    #fit in log space
    logQ = np.log(recession_Q + 1e-12) # add small value to avoid log(0)
    #use pwlf to fit piecewise linear function to logQ and extract slopes as recession coefficients
    pwlf_model = pwlf.PiecewiseLinFit(recession_t, logQ) # add small value to avoid log(0)
    t_segments = pwlf_model.fit(n_coeffs)
    slopes = pwlf_model.slopes 
    # linear fit to logQ 
    return slopes, t_segments

def main():
    # parser = ArgumentParser()
    # parser.add_argument('--input_data', type = str, help = "Path to npz file")
    # parser.add_argument('--output_path', type = str, help = "Output file path")
    # args = parser.parse_args()
    # input_data = args.input_data
    # output_path = args.output_path
    return



if __name__ == "__main__":
    main()

    