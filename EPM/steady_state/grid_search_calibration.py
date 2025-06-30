# import python packages
import numpy as np #for numerical operations
import pandas as pd #for handling dataframes
import sys
import os
import math
from mpi4py import MPI
from itertools import product 
import tqdm
import argparse
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src"))) #use this to be able to import local packages

from utils.config import *
from utils.utils import *
from utils.creeks import *
from utils.calibration import * 

#param space, specify here in dict form
def grid_search_calibration(run_data_dir, run_data_fname, sim_dir, max_runs = 1000):
    param_space = {
        'Kh_0' : list(np.arange(0.1, 10.1, 0.1)),
        'Kh_1' : list(np.arange(0.1, 5.1, 1)),
        'Kv_0' : list(np.arange(0.1, 10.1, 0.1)),
        'Kv_1' : list(np.arange(0.1, 5.1, 0.1)),
        'C_spring' : list(np.arange(0.1, 10.1, 0.1)),
        'C_creek' : list(np.arange(.1, 1.1, .1)),
    }


    #mpi configs
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # rank = 1
    # size = 1000

    print(f'rank {rank} started', flush = True)

    #generate valid combinations 
    values = list(param_space.values())
    keys = list(param_space.keys())
    filtered_combos = (
        combo for combo in (
            dict(zip(keys, values))
            for values in product(*values)
        )  
    )

    combo_gen = (
        combo
        for i, vals in enumerate(product(*values))
        if (i % size == rank)
        and (lambda combo: combo['Kh_1'] <= combo['Kh_0'] and combo['Kv_1'] <= combo['Kv_0'] and combo['Kh_0']== combo['C_spring'])(dict(zip(keys, vals)))
        # returns the dict only if it passes the condition
        for combo in [dict(zip(keys, vals))]
    )

    #set up model run base
    run = Config('EPM_2layer.yaml')
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
    run.extract_idomain()
    run.extract_creek_cells()

    if rank == 0:
        print('config complete', flush = True)
    #calibration data 
    mrsw = CalibrationData(name = 'mrsw', filename = '../../data/MRSW/MRSW_head_CSV.csv', UTME = 557091, UTMN = 4867265)
    bs_q = CalibrationData(name = 'bs_q', filename = '../../data/discharge/discharge_2017_2020.csv')

    mrsw.convert_data_to_timeseries()
    bs_q.convert_data_to_timeseries(datetime_col='date')

    mrsw.set_cal_value(mrsw.data['gw_elev[m]'].mean())
    bs_q.set_cal_value(bs_q.data.rolling(window = '7D').mean().mean()['Q [m3/s]'])

    mrsw_cell_idx = run.get_cell_id_from_coords(mrsw.UTME, mrsw.UTMN)
    bs_UTME, bs_UTMN = run.spring[run.spring.ID == '55A00572'].UTME, run.spring[run.spring.ID == '55A00572'].UTMN
    bs_cell_idx = run.get_cell_id_from_coords(bs_UTME, bs_UTMN)

    #performance tracking 
    run_data = pd.DataFrame(columns = keys + ['success', 'mrsw_head', 'mrsw_error', 'bs_q', 'bs_error', 'head_above_surface_error'])


    #grid search
    i = 0
    for combo in combo_gen:        
        run_name = f'{sim_dir}/creeks_{combo["C_creek"]}_springs_{combo["C_spring"]}_Kh_{combo["Kh_0"]}_{combo["Kh_1"]}_Kv_{combo["Kv_0"]}_{combo["Kv_1"]}'
        if rank == 0:
            print(run_name, flush = True)
        if os.path.exists(run_name):
            print(f'run {combo} already completed, skipping')
            continue
        else: 
            if i >= max_runs:
                break
            else:
                i += 1
                run.Kh = [combo['Kh_0'], combo['Kh_1']]
                run.Kv = [combo['Kv_0'], combo['Kv_1']]
                run.extract_K_values()
                for drain in run.drain_data:
                    if drain['name'] == 'creek':
                        drain['C'] = combo['C_creek']
                    elif drain['name'] == 'spring':
                        drain['C']= combo['C_spring']
                drn_spd = run.extract_drain_spd()
                run.make_sim(lenuni = "METER", ws = run_name)
                run.add_npf_module()
                run.add_recharge_module()
                run.add_drains_module()
                success, buff = run.run_sim()
                if success: 
                    run.read_head_output()
                    run.read_drain_discharge_output()

                    results = combo
                    results['success'] = True
                    results['mrsw_head'] = run.heads[0][mrsw_cell_idx]
                    results['mrsw_error'] = mrsw.get_residual(results['mrsw_head'])
                    results['bs_q'] = run.drain_array[0][bs_cell_idx]
                    results['bs_error'] = bs_q.get_residual(results['bs_q'])
                    results['head_above_surface_error'] = run.check_head_above_land_surface(return_type = 'count')

                else:
                    results = combo
                    results['success'] = False
                    results['mrsw_head'] = np.nan
                    results['mrsw_error'] = np.nan
                    results['bs_q'] = np.nan
                    results['bs_error'] = np.nan
                    results['head_above_surface_error'] = np.nan
                    for f in os.listdir(run_name): #delete files to save space
                        os.remove(f'{run_name}/{f}')
                run_data = pd.concat([run_data, pd.DataFrame(results)], axis = 0, ignore_index = True)
                print(f'{run_name} done')
    try: 
        run_data.to_csv(f'{run_data_dir}/{run_data_fname}_{rank}.csv', index = False)
        if rank == 0:
            print(f'data saved to {run_data_fname}', flush = True)
    except OSError:
        os.makedirs(run_data_dir)
        run_data.to_csv(f'{run_data_dir}/{run_data_fname}_{rank}.csv', index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    max_runs = parser.add_argument('max_runs', type = int, help = 'max runs per rank', default = 5)
    run_data_dir = parser.add_argument('data_output_dir', type = str, help = 'output dir for run data', default = 'EPM_2layer')
    run_data_fname = parser.add_argument('data_output_fname', type = str, help = 'file name pattern for run data csvs', default = 'results')
    sim_dir = parser.add_argument('sim_dir', type = str, help = 'path for model simulations', default = 'model_runs')
    grid_search_calibration(run_data_dir, run_data_fname, sim_dir, max_runs)









