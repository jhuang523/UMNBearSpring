import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src"))) #use this to be able to import local packages
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../src"))) #use this to be able to import local packages

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src"))) #use this to be able to import local packages
import numpy as np
import pandas as pd
import pickle
from utils.openkarst_network import OpenKarstNetwork as OKN
from utils.common import load_yaml, print_verbose, load_pickle, write_pickle
from run_openkarst import run_from_yaml
from argparse import ArgumentParser
from mpi4py import MPI


def run_openkarst_mpi(sim_list, verbose = False): 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0: # broadcast the input data files
        comm.bcast(sim_list) 
    input_data_file = sim_list[rank]
    run_from_yaml(input_data_file, verbose = verbose)

def main():
    parser = ArgumentParser(description="Run OpenKarst simulation on a given network")
    parser.add_argument("--sim_list_file", type = str, help = "File containing list of input data yamls to run")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()
    sim_list_file = args.sim_list_file
    verbose = args.verbose
    with open(sim_list_file) as f:
        sim_list = [line.strip() for line in f if line.strip()]
    print_verbose(f"simulation list: {sim_list}", verbose)
    run_openkarst_mpi(sim_list, verbose)


if __name__ == "__main__":
    main()






