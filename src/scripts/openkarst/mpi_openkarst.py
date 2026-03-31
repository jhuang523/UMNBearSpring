import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src"))) #use this to be able to import local packages
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../src"))) #use this to be able to import local packages

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src"))) #use this to be able to import local packages
import numpy as np
import pandas as pd
import pickle
import time
from utils.openkarst_network import OpenKarstNetwork as OKN
from utils.common import load_yaml, print_verbose, load_pickle, write_pickle
from run_openkarst import run_from_yaml
from argparse import ArgumentParser
from mpi4py import MPI


def run_openkarst_mpi(sim_list, log_path, verbose=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # rank 0 already has sim_list, broadcast to everyone else
    sim_list = comm.bcast(sim_list, root=0)

    if rank == 0:
        # logger — one start + one end message per simulation per worker rank
        num_workers = size - 1
        expected_messages = len(sim_list) * 2  # start + end per sim

        with open(log_path, "w") as f:
            messages_received = 0
            while messages_received < expected_messages:
                msg = comm.recv(source=MPI.ANY_SOURCE, tag=0)
                f.write(msg)
                f.flush()
                messages_received += 1
    else:
        worker_rank = rank - 1
        num_workers = size - 1
        sub_list = sim_list[worker_rank::num_workers]

        for input_data_file in sub_list:
            start = time.strftime("%Y-%m-%d %H:%M:%S")
            comm.send(f"[Rank {rank:03d}] START: {input_data_file} at {start}\n", dest=0, tag=0)
            try: 
                run_from_yaml(input_data_file, verbose=verbose)
                end = time.strftime("%Y-%m-%d %H:%M:%S")
                comm.send(f"[Rank {rank:03d}] END:   {input_data_file} at {end}\n", dest=0, tag=0)
            except Exception as e:
                end = time.strftime("%Y-%m-%d %H:%M:%S")
                comm.send(f"[Rank {rank:03d}] ERROR: {input_data_file} at {end} with error {e}\n", dest=0, tag=0)

def main():
    parser = ArgumentParser(description="Run OpenKarst simulation on a given network")
    parser.add_argument("--sim_list_file", type = str, help = "File containing list of input data yamls to run")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument("--log_path", type = str, help = "path for outlet logging" )

    args = parser.parse_args()
    sim_list_file = args.sim_list_file
    verbose = args.verbose
    log_path = args.log_path
    with open(sim_list_file) as f:
        sim_list = [line.strip() for line in f if line.strip()]
    print_verbose(f"simulation list: {sim_list}", verbose)
    run_openkarst_mpi(sim_list, log_path, verbose)


if __name__ == "__main__":
    main()






