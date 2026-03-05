# import openkarst
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src"))) #use this to be able to import local packages
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../src"))) #use this to be able to import local packages

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src"))) #use this to be able to import local packages
import numpy as np
from utils.common import write_pickle
from argparse import ArgumentParser

def check_convergence(X, tol = 1e-6):
    #X is an nt x nx array. Checks for convergence at the last two timesteps
    if np.all(np.abs(X[-1] - X[-2])) > tol:
        return False
    return True 
def check_Q(Q, tol = 1e-6):
    #check convergence
    #TODO: additional quality checks? 
    return check_convergence(Q, tol= tol)  

def check_h(h, tol = 1e-6):
    return check_convergence(h, tol = tol) and np.all(h >= 0)
    
def extract_steady_state_conditions(results_npz):
    results = np.load(results_npz)
    Q = results["Q"]
    h = results["y"]
    #some QC 
    if not check_Q(Q):
        raise Exception("Q not converged")
    if not check_h(h):
        raise Exception("h not converged")
    #
    Q_ss = Q[-1]
    h_ss = h[-1]
    IC = {'initial_flowrate' : Q_ss, 'initial_water_depth' : h_ss}
    return IC

def main():
    parser = ArgumentParser()
    parser.add_argument('--input_data', type = str, help = "Path to npz file")
    parser.add_argument('--output_path', type = str, help = "Output file path")
    args = parser.parse_args()
    input_data = args.input_data
    output_path = args.output_path
    IC = extract_steady_state_conditions(input_data)
    write_pickle(output_path, IC)


if __name__ == "__main__":
    main()

    