# import openkarst
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src"))) #use this to be able to import local packages
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../src"))) #use this to be able to import local packages

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src"))) #use this to be able to import local packages
import numpy as np
from utils.common import write_pickle
from argparse import ArgumentParser

def relative_L2_norm(X):
    return np.linalg.norm(np.diff(X, axis = 0), axis = 0)/(np.linalg.norm(X[:-1], axis = 0) + 1e-12)
def check_convergence(X, tol = 1e-3, n_steps = 10):
    """Check using relative error"""
    #X is an nt x nx array or a 1 x nt array. Checks for convergence at the last n_steps timesteps
    # if np.mean(np.abs(np.diff(X[-(n_steps+1):], axis = 0)/X[-(n_steps+1):-1])) < tol:
    if (relative_L2_norm(X[-(n_steps+1):])).mean() < tol:
        return True
    return False

def check_Q(Q, tol = 1e-3, n_steps = 10):
    #check convergence
    #TODO: additional quality checks? 
    convergence = check_convergence(Q, tol= tol, n_steps= n_steps)
    real = np.isreal(Q).all()
    if not convergence:
        print("No convergence")
    if not real:
        print("Q contains non-real values")
    return convergence and real

def check_h(h, tol = 1e-3, n_steps = 10):
    convergence = check_convergence(h, tol= tol, n_steps= n_steps)
    real = np.isreal(h).all()
    positive = np.all(h >= 0)
    if not convergence:
        print("No convergence")
    if not real:
        print("h contains non-real values")
    if not positive:
        print("h contains negative values")
    return convergence and real and positive

def extract_steady_state_conditions(results_npz, Q_tol = 1e-3, h_tol = 1e-3, n_steps = 10):
    results = np.load(results_npz)
    Q = results["Q"]
    h = results["y"]
    #some QC 
    if not check_Q(Q, tol = Q_tol, n_steps = n_steps):
        print("Q not converged")
        if not check_h(h, tol = h_tol, n_steps = n_steps):
            print("h not converged")
            raise ValueError("Neither Q nor h converged. Cannot extract steady state conditions.")
        raise ValueError("Q not converged. Cannot extract steady state conditions.")
    
    #
    Q_ss = Q[-1]
    h_ss = h[-1]
    IC = {'initial_flowrate' : Q_ss, 'initial_water_depth' : h_ss}
    return IC

def spring_flow(network, Q): 

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

    