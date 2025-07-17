import numpy as np #for numerical operations
import pandas as pd #for handling dataframes
import sys
import os
import time
import argparse
import yaml
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src"))) #use this to be able to import local packages

from utils.config import *
from utils.utils import *
from utils.creeks import *
from utils.calibration import * 

def main(config_file, verbose = False, transient = False, **model_params):
    """run model, compare to calibration points, export data"""
    start_time = time.time()
    run = Config(config_file)
    run.update_params(**model_params)
    run.load_polygon('watershed', 'springshed', 'subdomain')
    run.merge_polygons('merged', 'watershed_polygon', 'springshed_polygon')
    run.load_creeks()
    run.set_domain('subdomain_polygon')
    run.apply_DEM_to_domain()
    start = run.creeks.return_coordinates(186)[-1]
    nearest = get_nearest_point(start, run.merged_polygon)
    run.creeks.extend_creek(start, nearest)
    start = run.creeks.return_coordinates(154)[-1]
    nearest = get_nearest_point(start, run.merged_polygon)
    run.creeks.extend_creek(start, nearest)
    run.creeks.clip_creek(run.domain, 10)
    run.load_karst_features()
    run.extract_grid_params_from_domain()
    run.extract_top_config()
    run.extract_bottom_config()
    run.create_grid()
    run.import_idomain()
    run.extract_creek_cells()
    springshed_cells = run.extract_polygon_cells(run.springshed_polygon)
    springshed_top = np.hstack([np.zeros((springshed_cells.shape[0], 1)), springshed_cells])
    springshed_botm = np.hstack([np.ones((springshed_cells.shape[0], 1)), springshed_cells])
    run.extract_K_values()
    run.set_K_values(springshed_top, Kh = run.Kh_ss[0], Kv = run.Kv_ss[0])
    run.set_K_values(springshed_botm, Kh = run.Kh_ss[1], Kv =run.Kv_ss[1])
    run.import_conduit_network(verbose = verbose)
    run.set_conduit_K_vals(verbose = verbose)
    drn_spd = run.extract_drain_spd()
    run.make_sim(lenuni = "METER")
    run.add_npf_module(icelltype = 1)
    if transient:
        run.extract_recharge()
    run.add_recharge_module()
    run.add_drains_module()
    results = {
        'ncol' : run.ncol,
        'nrow' : run.nrow,
        'nlay' : run.nlay,
        'n_idomain' : np.argwhere(run.idomain == 1).shape[0],
        'n_springshed_cells' : springshed_cells.shape[0],
        'n_conduit_cells' : np.argwhere(run.network == 1).shape[0],
        'Kh' : run.Kh,
        'Kv' : run.Kv,
        'Kh_ss' : run.Kh_ss,
        'Kv_ss' : run.Kv_ss,
        'Kh_conduit' : run.Kh_conduit,
        'Kv_conduit' : run.Kv_conduit,
        'drain_data' : run.drain_data,
        'network_file' : run.npy['conduit_network'],
        'idomain_file' : run.npy['idomain'],
        'recharge' : run.rech
    }
    if transient:
        results['ss'] = run.ss
        results['sy'] = run.sy

    if not run.validate_config():
        raise ValueError("Validation failed. Exiting simulation")
    success, buff = run.run_sim()
    results['success'] = success
    end_time = time.time()
    run_time = end_time - start_time
    results['run_time'] = run_time
    # print_output('calibration data loaded', zero_only=True)
    #performance tracking 
    run_data_path = f'{run.ws}/run_data.yaml'
    with open(run_data_path, 'w') as file:
        for k, v in results.items():
            file.write(f"{k}: {v}\n")
        print_verbose('run data saved', verbose)
    
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type = str, help = 'settings file path')
    parser.add_argument('--verbose', action = 'store_true')
    parser.add_argument('--transient', action = 'store_true')
    parser.add_argument("--params", nargs='*', help="key=value pairs")
    args = parser.parse_args()
    model_params = dict(param.split("=", 1) for param in args.params or [])
    main(args.config_file, verbose=args.verbose, transient= args.transient, **model_params)


    