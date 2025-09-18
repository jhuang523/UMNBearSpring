# import python packages
import numpy as np #for numerical operations
from matplotlib.ticker import FormatStrFormatter
import pandas as pd #for handling dataframes
import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src"))) #use this to be able to import local packages

from utils.config import *
from utils.utils import *
from utils.creeks import *
from utils.calibration import * 
import scripts.run_model as run_model
import scripts.check_outputs as check_outputs

def main(config_file, max_iter = 10, ws = 'calibration', conduits = True, run_name = 'run', alpha = 1):
    def dict_less_than(dict1, dict2):
        if dict1 is None or dict2 is None:
            return False 
        return all(abs(dict1.get(k))< dict2.get(k) for k in dict1.keys() & dict2.keys())
    khkv_ratio = 0.1
    error_threshold = {'head' : 0.01, 
                    'peak_flow' : 0.01, 
                    'base_flow' : 0.01, 
                    'head_above_surface': 0.01
    }
    errors = {'head': np.nan,
            'peak_flow' : np.nan,
            'base_flow' : np.nan,
            'head_above_surface': np.nan}
    model_params = {}
    calibration_data = []
    i = 0

    while i < max_iter and not dict_less_than(errors, error_threshold):
        model_params['ws'] = f"{ws}/{run_name}_{i}"
        run, success = run_model.main(config_file = config_file, transient = True, conduits = conduits, **model_params)
        errors = {'head': np.nan,
            'peak_flow' : np.nan,
            'base_flow' : np.nan,
            'head_above_surface': np.nan}
        calibration_run_data = {'Kh' : run.Kh, 
                'Kv' : run.Kv,
                'Kh_ss' : run.Kh_ss,
                'Kv_ss' : run.Kv_ss,
                'Kh_conduit' : run.Kh_conduit,
                'Kv_conduit' : run.Kv_conduit,
                'drain_data' : run.drain_data}
        if success: 
            check_outputs.main(config_file=config_file, **model_params)
            model_params['drain_data'] = run.drain_data
            for obj in run.calibration_data: 
                if obj['name'] == 'head_above_surface':
                    run_data = load_yaml(f'{run.ws}/run_data.yaml')
                    errors['head_above_surface'] = run_data['head_above_surface_error']/run_data['n_idomain']
                else:
                    data = pd.read_csv(f"{run.ws}/{obj['name']}_{obj['type']}.csv")
                    data = index_to_date(data, data.columns[0])
                    if obj['type'] == 'head': 
                        errors['head'] = data.error.mean()/((data.total - data.error).mean())
                    elif obj['type'] == 'discharge': 
                        baseflow_cutoff = (data.total - data.error).rolling(window = '7D').mean().mean()
                        baseflow = data[data.total <= baseflow_cutoff]
                        peakflow = data[data.total > baseflow_cutoff]
                        errors['base_flow']= baseflow.error.mean()/(baseflow.total - baseflow.error).mean()
                        errors['peak_flow'] = peakflow.error.mean()/(peakflow.total - peakflow.error).mean()
            if abs(errors['head']) > error_threshold['head']:
                model_params['Kh'] = [max(k * (1 - errors['head'] * alpha), 0)for k in run.Kh]
                model_params['Kv'] = [max(k * khkv_ratio,0) for k in model_params['Kh']]
            if abs(errors['base_flow']) > error_threshold['base_flow']:
                model_params['Kh_ss'] = [max(k * (1 - errors['base_flow'] * alpha),0) for k in run.Kh_ss]
                model_params['Kv_ss'] = [max(k * khkv_ratio, 0) for k in model_params['Kh_ss']]
            if abs(errors['peak_flow']) > error_threshold['peak_flow']:
                model_params['Kh_conduit'] = [max(k * (1 - errors['peak_flow'] * alpha),0) for k in run.Kh_conduit]
                model_params['Kv_conduit'] = [max(k * khkv_ratio,0) for k in model_params['Kh_conduit']]
                model_params['drain_data']['spring'] = model_params['Kh_conduit'][0]
            if abs(errors['head_above_surface']) > error_threshold['head_above_surface']:
                C_creek = max(run.drain_data['creek'] * (1 + errors['head_above_surface'] * alpha), 0)
                model_params['drain_data']['creek'] =  C_creek
        write_yaml(f'{run.ws}/run_data.yaml', {'calibration' :errors})        
        calibration_run_data = calibration_run_data | errors
        print(calibration_run_data)
        calibration_data.append(calibration_run_data)
        i+=1
    calibration_data = pd.DataFrame(calibration_data)
    calibration_data.to_csv('calibration_data.csv', index = False)
    return calibration_data
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--max_iter', type = int, default = 10)
    parser.add_argument('--alpha', type = float, default = 1)
    parser.add_argument('--ws', default = 'calibration')
    parser.add_argument('--run_name', default = 'run')
    parser.add_argument('--no_conduits', action= 'store_false')
    args = parser.parse_args()
    main(config_file=args.config_file, max_iter=args.max_iter, ws = args.ws, alpha = args.alpha, run_name= args.run_name, conduits = True if args.no_conduits is not None else False)



        

