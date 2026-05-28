import geopandas as gpd
import shapely as shp
import numpy as np
import pandas as pd 
import yaml
import rasterio
import os
from utils.geos import *
import pickle


def load_yaml(path : str):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def write_yaml(path: str, data : dict):
    with open(path, 'a') as file:
        for k, v in data.items():
            file.write(f'{k}: {v}')

def load_pickle(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pickle(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
def convert_df_to_datetime(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col])
    return df

def convert_hourly_to_daily(df, date_col, agg = 'mean'):
    non_date_cols = list(df.columns)
    non_date_cols.remove(date_col)
    df = convert_df_to_datetime(df, date_col).groupby(df[date_col].dt.date)[non_date_cols].agg(agg)
    df.index = pd.to_datetime(df.index)
    return df

def index_to_date(df, date_col = 'DATE'):
    """sets index to date col"""
    df = convert_df_to_datetime(df, date_col)
    df.index = df[date_col]
    return df.drop(date_col, axis = 1)

def print_verbose(string, verbose):
    if verbose:
        print(string)