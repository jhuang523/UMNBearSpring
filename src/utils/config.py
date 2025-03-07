# helper functions go here
import numpy as np
import geopandas as gpd
import pandas as pd
import yaml
import flopy
from utils.utils import *

class Config:
    """reads config info from yaml file"""
    def __init__(self, yaml_file, **config):
        config = load_yaml(yaml_file)
        self.__dict__.update(**config)
        print(f"new config loaded from {yaml_file}")
        self.load_geojsons()
        self.load_csvs()
    def __repr__(self):
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
        """Takes polygon and assigns it as domain"""
        self.domain = getattr(self, polygon_id)
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
        
    def write_sim(self): 
        pass 

    

