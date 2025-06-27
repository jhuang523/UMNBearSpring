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
    polygon = polygon_data['geometry'].apply(shp.make_valid)
    if polygon.is_valid.iloc[0]:
        return polygon.geometry.iloc[0]
    else:
        print('polygon not valid, check data and try again')
        return None
def combine_polygons(*polygons, crs = 'EPSG:32615'): 
    """ Takes 2 polygons and returns a merged polygon that covers the max extent of both"""
    return shp.ops.unary_union(polygons)


def apply_DEM_to_polygon(dem_path : str, polygon_data, delr, delc, crs = 'EPSG:32615'):
    """Given DEM_path, polygon obj. and delr and delc, creates array of elevation values for the polygon. returns dem_grid"""
    if isinstance(polygon_data, shp.geometry.Polygon): 
        bounds = gpd.GeoDataFrame(geometry = [polygon_data]).total_bounds 
    elif isinstance(polygon_data, gpd.GeoDataFrame):
        bounds = polygon_data.total_bounds
    width = int((bounds[2] - bounds[0])/delr)
    height = int((bounds[3] - bounds[1])/delc)
    with rasterio.open(dem_path) as src:
        #define the window of useful data(in UTM coords) from the larger DEM shapefile
        window = rasterio.windows.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], transform=src.transform)
        extent = rasterio.windows.bounds(window, src.transform)#get window boundaries to confirm they are correct
        print(extent)
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
def get_point(coords : tuple) -> shp.geometry.Point:
    return shp.geometry.Point(coords)

def get_nearest_point(geom1, geom2) -> shp.geometry.Point:
    """Return the nearest point on geom2 to geom1. If a tuple is passed for either arg, then converts to a point"""
    if type(geom1) == tuple:
        geom1 = get_point(geom1)
    if type(geom2) == tuple:
        geom2 = get_point(geom2)
    elif isinstance(geom1, shp.geometry.Polygon):
        geom1 = geom1.boundary
    elif isinstance(geom2, shp.geometry.Polygon):
        geom2 = geom2.boundary
    return shp.ops.nearest_points(geom1, geom2)[1]

def convert_cell_id_to_coordinates(idx_list : list, delr, delc, **grid_lims):
    idx_list = idx_list.copy()
    if 'xmin' in grid_lims:
        b_x = grid_lims.get('xmin')
        a_x = 1
    elif 'xmax' in grid_lims:
        b_x = grid_lims.get('xmax')
        a_x = -1
    else: 
        print('no x lim provided, please provide xmin or xmax')
        exit
    if 'ymin' in grid_lims:
        b_y = grid_lims.get('ymin')
        a_y = 1
    elif 'ymax' in grid_lims:
        b_y = grid_lims.get('ymax')
        a_y = -1
    else:
        print('no y lim provided, please provide xmin or xmax')
        exit
    idx_list[:,1] = idx_list[:,1]*delc * a_x + b_x 
    idx_list[:,0] =  idx_list[:,0]  *delr * a_y + b_y
    idx_list[:, [0,1]] = idx_list[:, [1, 0]]
    return idx_list
        
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

def get_cell_id_from_coords(x, y, x0, ymax, dx, dy): 
    j = ((x-x0)/dx) + 1
    i = ((ymax-y)/dy) 
    return i.astype(int), j.astype(int)

def convert_df_to_datetime(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col])
    return df

def index_to_date(df, date_col = 'DATE'):
    """sets index to date col"""
    df = convert_df_to_datetime(df, date_col)
    df.index = df[date_col]
    return df.drop(date_col, axis = 1)