import geopandas as gpd
import shapely as shp
import numpy as np
import pandas as pd 
import yaml
import rasterio
import os

def load_yaml(path : str):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_geojson(path : str, crs = 'EPSG:32615'):
    return gpd.read_file(path).to_crs(crs)

def extract_polygon(geodf):
    """Takes geodataframe (already projected), extracts and returns polygon data"""
    polygon_data = geodf[geodf.geometry.type == 'Polygon']
    polygon_data.loc[:,'geometry'] = polygon_data['geometry'].apply(shp.make_valid)
    if polygon_data.is_valid.iloc[0]:
        return polygon_data.explode(index_parts = True)
    else:
        print('polygon not valid, check data and try again')
        return None
def combine_polygons(*polygons, crs = 'EPSG:32615'): 
    """ Takes 2 polygons and returns a merged polygon that covers the max extent of both"""
    all_edges = []
    for p in polygons:
        all_edges.append(p['geometry'])
    return gpd.GeoDataFrame(geometry = [shp.ops.unary_union(all_edges)], crs = crs)


def apply_DEM_to_polygon(dem_path : str, polygon, delr, delc):
    """Given DEM_path, polygon obj. and delr and delc, creates array of elevation values for the polygon. returns dem_grid"""
    bounds = polygon.total_bounds 
    width = int((bounds[2] - bounds[0])/delr)
    height = int((bounds[3] - bounds[1])/delc)
    with rasterio.open(dem_path) as src:
        #define the window of useful data(in UTM coords) from the larger DEM shapefile
        window = rasterio.windows.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], transform=src.transform)
        extent = rasterio.windows.bounds(window, src.transform)#get window boundaries to confirm they are correct
        
        #convert all of the data to a grid so it can be used with the MODFLOW model
        # transform = rasterio.transform.from_bounds(*extent, width, height) #transform the data to the grid #TODO: what is this used for? 
        
        #Import the DEM and apply it to the grid
        dem_grid = src.read(
        1,
        out_shape = (height, width),
        window = window,
        resampling= rasterio.enums.Resampling.bilinear
        )
        
        #mask out the erroneous data (excessively large values due to data errors)
        maxval = 10000
        dem_grid = np.ma.masked_where(dem_grid > maxval, dem_grid)
        return dem_grid

def bres(row1, col1, row2, col2):
    """Generate cell indices along a straight line in a 2D grid using Bresenham's algorithm."""
    cells = []
    d_row = abs(row2 - row1)
    d_col = abs(col2 - col1)
    sign_row = 1 if row2 > row1 else -1
    sign_col = 1 if col2 > col1 else -1
    err = d_col - d_row

    while (row1 != row2 or col1 != col2):
        cells.append((row1, col1))
        err2 = 2 * err
        if err2 > -d_row:
            err -= d_row
            col1 += sign_col
        if err2 < d_col:
            err += d_col
            row1 += sign_row

    cells.append((row2, col2))  # Add the last cell
    return cells
