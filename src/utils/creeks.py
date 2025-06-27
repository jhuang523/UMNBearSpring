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
    def remove_segment(self, *FID):
        self.data = self.data[self.data['FID'] not in FID]
        print(f'removed segments {FID}')
    def extend_creek(self, start, end): 
        """take in start and end coordinates (x, y) and draw a straight line from start to end. add linestring to self.data"""
        if type(start) == tuple:
            start = get_point(start)
            print('converted start tuple to Point')
        if type(end) == tuple:
            end = get_point(end)
            print('converted end tuple to Point')
        new_segment = shp.geometry.LineString([start, end])
        self.add_new_segment(new_segment)
        print(f'added segment from {start} to {end}')
        print(self)
    def clip_creek(self, geom, tolerance : float):
        """Removes segments outside of given geom. Tolerance is given in absolute value"""
        self.data = gpd.clip(self.data, geom.buffer(tolerance))
        print("clipped creeks")
    def return_coordinates(self, FID):
        return self.return_segment(FID).geometry.coords
    def map_creek_to_grid(self, nrow, ncol, delr, delc, xmin, ymax):
        creek_lines = list(self.data.geometry)
        rasterized = features.rasterize(
            [(line, 1) for line in creek_lines],  # Assign value 1 where the line intersects
            out_shape=(nrow, ncol),
            transform=rasterio.transform.from_origin(xmin, ymax, delc, delr),
            all_touched=True  # Ensures partial cells are included
        )
        intersecting_cells = np.argwhere(rasterized == 1)
        return intersecting_cells
    

    