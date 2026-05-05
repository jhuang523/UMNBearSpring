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
    df['run_id'] = run_id if run_id is not None else float(time.time())
    if method == "parquet":
        df.to_parquet(output_path, partition_cols=partition, compression='snappy')
        print_verbose(f"Outlet flow written to {output_path} with run_id {df['run_id'].iloc[0]}", verbose)
    else:
        #TODO: Add other output methods
        raise ValueError("Unsupported output method. Use 'parquet'.")


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_data', type = str, help = "Path to npz file")
    parser.add_argument('--output_path', type = str, help = "Output file path")
    args = parser.parse_args()
    input_data = args.input_data
    output_path = args.output_path



if __name__ == "__main__":
    main()

    