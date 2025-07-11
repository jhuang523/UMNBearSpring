import pandas as pd
import numpy as np
import shapely as shp
import matplotlib.pyplot as plt 
import flopy as mf
from utils.utils import *
from rasterio import features

class Creek:
    def __init__(self, path = None, geodf = None):
        if path == None:
            self.data = geodf
        else:
            self.data = load_geojson(path)
        self.crs = self.data.crs
    def __repr__(self):
        self.data.plot()
        return""
    def plot(self, ax = None, **params):
        self.data.plot(ax = ax, **params)
    def return_segment(self, FID):
        """return segment based on FID"""
        return self.data[self.data['FID'] == FID].iloc[0]
    def add_new_segment(self, segment: shp.geometry.LineString):
        new_gpd = gpd.GeoDataFrame({'FID' :[max(self.data['FID']) + 1]}, geometry = [segment], crs = self.crs)
        self.data = pd.concat([self.data, new_gpd], axis = 0, ignore_index = True)
    def remove_segment(self, verbose = False, *FID):
        self.data = self.data[self.data['FID'] not in FID]
        print_verbose(f'removed segments {FID}', verbose)
    def extend_creek(self, start, end, verbose = False): 
        """take in start and end coordinates (x, y) and draw a straight line from start to end. add linestring to self.data"""
        if type(start) == tuple:
            start = get_point(start)
            print_verbose('converted start tuple to Point', verbose)
        if type(end) == tuple:
            end = get_point(end)
            print_verbose('converted end tuple to Point', verbose)
        new_segment = shp.geometry.LineString([start, end])
        self.add_new_segment(new_segment)
        print_verbose(f'added segment from {start} to {end}', verbose)
        print_verbose(self, verbose)
    def clip_creek(self, geom, tolerance : float, verbose = False):
        """Removes segments outside of given geom. Tolerance is given in absolute value"""
        self.data = gpd.clip(self.data, geom.buffer(tolerance))
        print_verbose("clipped creeks", verbose)
    def return_coordinates(self, FID):
        return self.return_segment(FID).geometry.coords
    def map_creek_to_grid(self, nrow, ncol, delr, delc, xmin, ymax):
        creek_lines = list(self.data.geometry)
        return map_geometry_to_grid(creek_lines, nrow, ncol, delr, delc, xmin, ymax)
    

    