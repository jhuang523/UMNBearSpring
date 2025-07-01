# import python packages
import numpy as np #for numerical operations
import pandas as pd #for handling dataframes
import sys
import os
import math
from itertools import product 
import tqdm
import time
import argparse
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src"))) #use this to be able to import local packages

from utils.config import *
from utils.utils import *
from utils.creeks import *
from utils.calibration import * 


    
    

#param space, specify here in dict form
def grid_search_calibration(run_data_dir, run_data_fname, sim_dir, max_runs = 1000, is_mpi = True, overwrite = False):
    def print_output(output, zero_only = False):
        if is_mpi:
            if zero_only == True and rank == 0:
                print(output, flush = True)
            elif zero_only == False:
                print(output, flush = True)



    param_space = {
        'Kh_0' : list(np.arange(0.1, 10.1, 1)),
        'Kh_1' : list(np.arange(0.1, 5.1, 1)),
        'Kv_0' : list(np.arange(0.1, 10.1, 1)),
        'Kv_1' : list(np.arange(0.1, 5.1, 1)),
        'Kh_0_ss' : list(np.arange(100, 550, 50)),
        'Kv_0_ss' : list(np.arange(100, 550, 50)),
        'Kh_1_ss' : list(np.arange(100, 550, 50)),
        'Kv_1_ss' : list(np.arange(100, 550, 50)),
        'C_spring' : list(np.arange(0.1, 10.1, 0.1)),
        'C_creek' : list(np.arange(.1, 1.1, .1)),
    }


    #mpi configs
    if is_mpi: 
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1000

    print_output(f'rank {rank} started')

    #generate valid combinations 
    values = list(param_space.values())
    keys = list(param_space.keys())

    combo_gen = (
        combo
        for i, vals in enumerate(product(*values))
        if (i % size == rank)
        and (lambda combo: combo['Kh_1'] < combo['Kh_0'] and combo['Kv_1'] < combo['Kv_0'] and combo['Kh_0']== combo['C_spring'] and combo['Kh_1_ss'] < combo['Kh_0_ss'] and combo['Kv_1_ss'] < combo['Kv_0_ss'])(dict(zip(keys, vals)))
        # returns the dict only if it passes the condition
        for combo in [dict(zip(keys, vals))]
    )

    #set up model run base
    print_output('setting up config', zero_only=True)
    run = Config('EPM_2layer.yaml')
    run.load_polygon('watershed', 'springshed', 'subdomain')
    run.merge_polygons('merged', 'watershed_polygon', 'springshed_polygon')
    run.load_creeks()
    run.set_domain('subdomain_polygon')
    run.apply_DEM_to_domain()
    print_output('domain set', zero_only=True)
    start = run.creeks.return_coordinates(186)[-1]
    nearest = get_nearest_point(start, run.merged_polygon)
    run.creeks.extend_creek(start, nearest)
    start = run.creeks.return_coordinates(154)[-1]
    nearest = get_nearest_point(start, run.merged_polygon)
    run.creeks.extend_creek(start, nearest)
    run.creeks.clip_creek(run.domain, 10)
    print_output('creeks set', zero_only=True)

    run.load_karst_features()
    run.extract_grid_params_from_domain()
    run.extract_top_config()
    run.extract_bottom_config()
    run.create_grid()
    print_output('grid set', zero_only=True)
    start = time.time()
    run.import_idomain()
    end = time.time()
    run_time = end - start
    print(f'time elapsed: {run_time:.3f} ')
    print_output('idomain set', zero_only=True)
    run.extract_creek_cells()
    springshed_cells = run.extract_polygon_cells(run.springshed_polygon)
    springshed_top = np.hstack([np.zeros((springshed_cells.shape[0], 1)), springshed_cells])
    springshed_botm = np.hstack([np.ones((springshed_cells.shape[0], 1)), springshed_cells])


    print_output('config complete', zero_only = True)
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
        print_output(run_name, zero_only = True)
        if os.path.exists(run_name) and overwrite == False:
            print(f'run {combo} already completed, skipping', flush = True)
            continue
        else: 
            if i >= max_runs:
                break
            else:
                i += 1
                run.Kh = [combo['Kh_0'], combo['Kh_1']]
                run.Kv = [combo['Kv_0'], combo['Kv_1']]
                run.extract_K_values()
                run.set_K_values(springshed_top, Kh = combo['Kh_0_ss'], Kv = combo['Kv_0_ss'])
                run.set_K_values(springshed_botm, Kh = combo['Kh_1_ss'], Kv = combo['Kv_1_ss'])

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
                run_data = pd.concat([run_data, pd.DataFrame([results])], axis = 0, ignore_index = True)
                print(f'{run_name} done', flush = True)
    try: 
        run_data.to_csv(f'{run_data_dir}/{run_data_fname}_{rank}.csv', index = False)
        if rank == 0:
            print(f'data saved to {run_data_fname}', flush = True)
    except OSError:
        print("making directory")
        os.makedirs(run_data_dir)
        run_data.to_csv(f'{run_data_dir}/{run_data_fname}_{rank}.csv', index = False)
        print(f'data saved to {run_data_fname}', flush = True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_runs', type = int, help = 'max runs per rank', default = 5)
    parser.add_argument('--data_output_dir', type = str, help = 'output dir for run data', default = 'EPM_2layer')
    parser.add_argument('--data_output_fname', type = str, help = 'file name pattern for run data csvs', default = 'results')
    parser.add_argument('--sim_dir', type = str, help = 'path for model simulations', default = 'model_runs')
    parser.add_argument('--no_mpi', action='store_false', dest='is_mpi', help='Disable MPI (default is enabled)')
    parser.add_argument('--overwrite', action='store_true', dest='overwrite', help='overwrite existing model runs')

    parser.set_defaults(is_mpi=True)    
    args = parser.parse_args()
    max_runs = args.max_runs
    run_data_fname = args.data_output_fname
    run_data_dir = args.data_output_dir
    sim_dir = args.sim_dir



    grid_search_calibration(run_data_dir, run_data_fname, sim_dir, max_runs, is_mpi = args.is_mpi, overwrite = args.overwrite)






