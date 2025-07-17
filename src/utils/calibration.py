import pandas as pd
import numpy as np
from utils.utils import *
from utils.creeks import * 
from utils.config import * 


class CalibrationData:
    """calibration data object. contains data, name, and coordinates, and cal_value"""
    def __init__(self, verbose = False,**params):
        self.name = params.get('name')
        data = params.get('data')
        filename = params.get('filename')
        self.UTMN = params.get('UTMN')
        self.UTME = params.get('UTME')
        skiprows = params.get('skiprows')
        self.load_data(data, filename, skiprows=skiprows)
        print_verbose(f'calibration obj {self.name} created', verbose)
    
    def __str__(self):
        return(f'{self.name}, {self.data}')

    def load_data(self, data = None, filename = None, verbose = False, **params): 
        skiprows = params.get('skiprows')
        if data is not None: #if is data itself and not filename 
            self.data = data
            print_verbose('set data',verbose)
        elif filename is not None: #read data
            if filename.endswith('.csv'): 
                self.data = pd.read_csv(filename, skiprows= skiprows)
                print_verbose(f'read {filename} and set data',verbose)
        else:
            print('no data read')
    def convert_data_to_timeseries(self, datetime_col = 'datetime', verbose = False):
        self.data = index_to_date(self.data, datetime_col)
        print_verbose('updated data to time series',verbose)

    def convert_data_to_daily(self, datetime_col = 'datetime'):
        self.data = convert_hourly_to_daily(self.data, date_col = datetime_col)

    def set_cal_value(self, cal_value = None, verbose = False):
        if cal_value is None:
            cal_value = self.data
        self.cal_value = cal_value 
        print_verbose(f'calibration value set to {cal_value}',verbose)

    def get_residual(self, val):
        if isinstance(self.cal_value, pd.DataFrame) and isinstance(val, pd.Series):
            residual = pd.DataFrame(pd.DataFrame(val).to_numpy() - self.cal_value.to_numpy())
            residual.index = self.cal_value.index
            return residual
        elif isinstance(self.cal_value, pd.Series) and isinstance(val, pd.DataFrame):
            residual = pd.DataFrame(val.to_numpy() - pd.DataFrame(self.cal_value).to_numpy())
            residual.index = self.cal_value.index
            return residual
        else:
            return val - self.cal_value
    
    def get_relative_residual(self, val):
        return self.get_residual(val)/self.cal_value

    def update_coordinates(self, UTME, UTMN, verbose=False):
        self.UTME = UTME
        self.UTMN = UTMN
        print_verbose(f'updated coordinates to {UTME}, {UTMN}',verbose)

