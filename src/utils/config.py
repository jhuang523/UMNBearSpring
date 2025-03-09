# helper functions go here
import numpy as np
import geopandas as gpd
import pandas as pd
import yaml
import flopy as mf
from utils.utils import *
from utils.creeks import * 

class Config:
    """reads config info from yaml file. contains domain and aquifer properties"""
    def __init__(self, yaml_file, **config):
        config = load_yaml(yaml_file)
        self.__dict__.update(**config)
        print(f"new config loaded from {yaml_file}")
        self.load_geojsons()
        self.load_csvs()
    def __repr__(self):
        try:
            self.domain.plot()
        except: 
            print("can't display domain")
        return str(vars(self).keys())
    def load_geojsons(self):
        """Load geojsons provided in yaml file and adds the data (GeoDataFrame) as attributes keyed by their key in the yaml. 
        i.e. watershed : "file_name" will read in file_name and can be accessed as self.watershed """
        geo_data = {}
        for id, path in self.geojson.items():
            geodf = load_geojson(path, crs = self.crs)
            geo_data[id] = geodf
            print(f'loaded {id} geojson')
        self.__dict__.update(**geo_data)
    def load_csvs(self):
        csv_data = {}
        for id, path in self.csv.items():
            csv = pd.read_csv(path)
            csv_data[id] = csv
            print(f'loaded {id} csv')
        self.__dict__.update(**csv_data)

    def load_polygon(self, *geodata_id):
        """Load polygons given a geodata attr. Returns as an attribute to self. 
        i.e. self.load_polygon('watershed') will take self.watershed and extract the polygon data from it.
         polygon data can be accessed using self.watershed_polygon """
        for id in geodata_id:
            geodf = getattr(self, id)
            polygon = extract_polygon(geodf)
            setattr(self, f'{id}_polygon', polygon)
            print (f"extracted {id} polygon")

    def load_external_polygon(self, polygon_id : str, polygon_data):
        """Take external polygon GeoDF and adds to self. keyed as {polygon_id}_polygon """
        setattr(self, f'{polygon_id}_polygon', polygon_data)
        print (f'added {polygon_id}')

    def merge_polygons(self, merged_polygon_id, *polygons):
        """Takes existing polygons via ID and adds a merged polygon to config keyed under id.
        E.g. self.merge_polygons('merged', 'watershed', 'springshed') will merge the watershed and springshed
        and key it under merged_polygon"""
        polygon_data = [getattr(self, name) for name in polygons]
        merged_polygon = combine_polygons(*polygon_data)
        setattr(self, f'{merged_polygon_id}_polygon', merged_polygon)
        print(f'merged {polygons}')

    def set_domain(self, polygon_id : str): 
        """Takes polygon and returns gdf """
        polygon =  getattr(self, polygon_id)
        self.domain = gpd.GeoDataFrame(geometry = [polygon], crs = self.crs)
        self.total_bounds = self.domain.total_bounds
        print(f'set domain to {polygon_id}')

    def apply_DEM_to_domain(self):
        try:
            self.dem_grid = apply_DEM_to_polygon(self.DEM, self.domain, self.delr, self.delc)
            print(f'applied DEM {self.DEM} to domain')
        except(FileNotFoundError):
            print('error, file not found')

    def load_karst_features(self):
        try:
            features = self.karst_features
            for feature_type in features['Type'].unique():
                setattr(self, feature_type, features[features['Type']==feature_type])
                print (f'added {feature_type}')
        except(KeyError) as e:
            print(e)        
    def load_creeks(self):
        if isinstance(self.creeks, Creek):
            print('you already have loaded creeks')
        elif isinstance(self.creeks, gpd.GeoDataFrame):
            self.creeks = Creek(geodf = self.creeks)
            print('loaded new Creek object from creeks data')
        else:
            print('error')
    def plot_polygons(self, *polygon_names, ax = None, **params):
        if ax == None:
            fig, ax = plt.subplots()
        polygons = [getattr(self, p) for p in polygon_names]
        gpd.GeoDataFrame(geometry = polygons, crs = self.crs).plot(ax = ax, **params)
    def extract_grid_params_from_domain(self): 
        """creates structured grid parameters based """
        self.Lx = self.total_bounds[2] - self.total_bounds[0]
        self.Ly = self.total_bounds[3] - self.total_bounds[1]
        self.ncol = int(self.Lx/self.delc)
        self.nrow = int(self.Ly/self.delr)
        self.nlay = len(self.botm)
        print(f"""extracted grid params: \nLx = {self.Lx}\nLy = {self.Ly}\nnrow = {self.nrow}\nncol = {self.ncol}\nnlay = {self.nlay}""")
    
    def extract_top_config(self):
        if self.top_config == 'DEM':
            self.top = self.dem_grid
            print ('set top to DEM')
        elif self.top_config == 'constant':
            self.top = self.top_elev
            print (f'set top to {self.top_elev}')
        #TODO: add other configurations for custom top array reading in data ETC 

    def extract_bottom_config(self):
        botm_elev = []
        for layer in self.botm:
            if layer['layer_type'] == 'constant_slope':
                azimuth = layer['azimuth']
                dip = np.radians(layer['dip'])
                slope = np.tan(dip)
                azimuth_rad = np.radians(azimuth)
                dx = self.delc
                dy = self.delr
                dz_dx = slope * np.cos(azimuth_rad)  # Change in Z per unit X
                dz_dy = slope * np.sin(azimuth_rad)  # Change in Z per unit Y
                x, y = np.meshgrid(np.arange(self.ncol), np.arange(self.nrow))  # Column (X) and Row (Y) indices
                # Compute bottom elevation, subtracting elevation change from max elevation
                if 'max_elev' in layer:
                    z0 = layer['max_elev']
                    layer_array = z0 - ((self.ncol - x-1)* dx * dz_dx) + ((self.nrow - y-1) * dy * dz_dy)
                elif 'min_elev' in layer:
                    z0 = layer['min_elev']
                    layer_array = z0 + (x * dx * dz_dx) + (y * dy * dz_dy) #TODO: adjust this 
                else:
                    print("no starting elevation given, cannot create array")
                    exit
                botm_elev.append(layer_array)
        self.botm_elev = np.stack(botm_elev, axis = 0)
        print(f'set bottom array')

    def create_grid(self, type = 'structured'):
        try:
            if type == 'structured':
                delr = np.ones(self.ncol, dtype = float) * self.delr
                delc = np.ones(self.nrow, dtype = float) * self.delc
                self.model_grid = mf.discretization.StructuredGrid(
                    lenuni = self.lenuni,
                    nlay=self.nlay,
                    nrow=self.nrow,
                    ncol=self.ncol,
                    delr=delr,
                    delc=delc,
                    top=self.top,
                    botm=self.botm_elev,
                    xoff=self.total_bounds[0],
                    yoff=self.total_bounds[1],
                    crs= self.crs
                )
            #TODO: configure for unstructured and other grid types 
        except KeyError as e: 
            print (f"error, missing grid param: {e}")
        except e:
            print(e)
    def plot_model_grid_layers(self, **params):
        azim = params.get('azim', None)
        cmap = params.get('cmap', 'viridis')
        elev = params.get('elev', None)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # y, x = np.meshgrid(np.arange(self.nrow) * self.delr + self.total_bounds[1],
        #            np.arange(self.ncol) * self.delc + self.total_bounds[0])
        x, y = np.meshgrid(np.arange(self.ncol) * self.delc + self.total_bounds[0], np.arange(self.nrow)* (-1 * self.delr) + self.total_bounds[3])
        z = np.copy(self.botm_elev)
        if hasattr(self, 'idomain'):
            x[self.idomain[0] == 0] = np.nan
            y[self.idomain[0] == 0] = np.nan
            z[self.idomain == 0] = np.nan
        for i in range(self.nlay):
            ax.plot_surface(x, y, z[i], cmap = cmap)
        ax.plot_surface(x, y, self.dem_grid)
        ax.view_init(azim = azim, elev = elev)
    def extract_idomain(self):
        xc, yc = self.model_grid.xcellcenters, self.model_grid.ycellcenters #Get the cell center coords
        points = np.array([(xc[i, j], yc[i, j]) for i in range(self.nrow) for j in range(self.ncol)])
        #Extract the raw polygon from the GDF object subDomain
        domain = self.domain.geometry.values[0]
        mask = shp.vectorized.contains(domain, points[:, 0], points[:, 1])
        idomain_mask = mask.reshape((self.nrow, self.ncol))
        idomain = np.zeros((self.nlay, self.nrow, self.ncol), dtype=int)
        idomain[:, :, :] = idomain_mask.astype(int)
        self.idomain = idomain
    def extract_K_values(self):
        Kh_array = []
        Kv_array = []
        for K in self.Kh:
            Kh_array.append(np.ones((self.ncol, self.nrow)) * K)
        self.Kh_vals = np.stack(Kh_array, axis = 0)
        for K in self.Kv:
            Kv_array.append(np.ones((self.ncol, self.nrow)) * K)
        self.Kv_vals = np.stack(Kv_array, axis = 0)
        #TODO deal with more variable K values
    # def assign_conduit_cells(self):
    #     return
    def write_sim(self): 
        pass 

    

