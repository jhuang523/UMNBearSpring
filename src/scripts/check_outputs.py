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

def main(config_file, verbose  = False, overwrite = False, **model_params):
    run = Config(config_file)
    run.update_params(**model_params)
    run.load_polygon('watershed', 'springshed', 'subdomain')
    run.merge_polygons('merged', 'watershed_polygon', 'springshed_polygon')
    run.load_creeks()
    run.load_karst_features()
    run.set_domain('subdomain_polygon')
    run.apply_DEM_to_domain()
    run.extract_grid_params_from_domain()
    run.load_sim()
    run.load_model()
    run.read_head_output()
    run.read_drain_discharge_output()
    results = {}
    for obj in run.calibration_data:
        if obj['name'] == 'head_above_surface':
            results['head_above_surface_error'] = run.check_head_above_land_surface(return_type = obj['type'])
            continue
        cal = CalibrationData(name = obj['name'], filename = obj['data'])
        cal.convert_data_to_timeseries(verbose=verbose)
        cal.set_cal_value(verbose=verbose)
        data = pd.DataFrame()
        for i in range(len(obj['UTME'])): 
            utme = obj['UTME'][i]
            utmn = obj['UTMN'][i]
            idx = run.get_cell_id_from_coords(utme, utmn)
            if obj['type'] == 'discharge': 
                temp = run.get_timeseries_discharge_single_cell(idx) * -1
            if obj['type'] == 'head': 
                temp = run.get_timeseries_head_single_cell(idx)
            temp.index = cal.cal_value.index
            data[f'{utme}, {utmn}'] = temp
        data['total'] = data[data.columns].sum(axis = 1)
        data['error'] = cal.get_residual(data['total'])
        cal_data_path = f'{run.ws}/{obj['name']}_{obj['type']}.csv'
        data.to_csv(cal_data_path)
        results[f'{obj['name']}_data'] = cal_data_path
    
    run_data_path = f'{run.ws}/run_data.yaml'
    mode = 'w' if overwrite else 'a'
    with open(run_data_path, mode) as file:
        for k, v in results.items():
            file.write(f"{k}: {v}\n")        
        print_verbose('output data saved', verbose)
    return results


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type = str, help = 'settings file path')
    parser.add_argument('--verbose', action = 'store_true')
    parser.add_argument('--overwrite', action= 'store_true')
    parser.add_argument("--params", nargs='*', help="key=value pairs")
    args = parser.parse_args()
    model_params = dict(param.split("=", 1) for param in args.params or [])
    main(args.config_file, verbose=args.verbose, overwrite = args.overwrite, **model_params)