import pandas as pd
import numpy as np
from utils.utils import *
from utils.creeks import * 
from utils.config import * 


class CalibrationData:
    """calibration data object. contains data, name, and coordinates, and cal_value"""
    def __init__(self, **params):
        self.name = params.get('name')
        data = params.get('data')
        filename = params.get('filename')
        self.UTMN = params.get('UTMN')
        self.UTME = params.get('UTME')
        skiprows = params.get('skiprows')
        self.load_data(data, filename, skiprows=skiprows)
        print(f'calibration obj {self.name} created')
    
    def __str__(self):
        return(f'{self.name}, {self.data}')

    def load_data(self, data = None, filename = None, **params): 
        skiprows = params.get('skiprows')
        if data is not None: #if is data itself and not filename 
            self.data = data
            print('set data')
        elif filename is not None: #read data
            if filename.endswith('.csv'): 
                self.data = pd.read_csv(filename, skiprows= skiprows)
                print(f'read {filename} and set data')
        else:
            print('no data read')
    def convert_data_to_timeseries(self, datetime_col = 'datetime'):
        self.data = index_to_date(self.data, datetime_col)
        print('updated data to time series')

    def convert_data_to_daily(self, datetime_col = 'datetime'):
        self.data = convert_hourly_to_daily(self.data, date_col = datetime_col)

    def set_cal_value(self, cal_value):
        self.cal_value = cal_value 
        print(f'calibration value set to {cal_value}')

    def get_residual(self, val):
        return val - self.cal_value
    
    def get_relative_residual(self, val):
        return self.get_residual(val)/self.cal_value

    def update_coordinates(self, UTME, UTMN):
        self.UTME = UTME
        self.UTMN = UTMN
        print(f'updated coordinates to {UTME}, {UTMN}')

