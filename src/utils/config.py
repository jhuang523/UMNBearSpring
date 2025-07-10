# helper functions go here
import numpy as np
import geopandas as gpd
import pandas as pd
import flopy as mf
import shapely as shp
from utils.utils import *
from utils.creeks import * 
from utils.calibration import *
import time

class Config:
    """reads config info from yaml file. contains domain and aquifer properties. stores all relevant model params + objects"""
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
    def load_geojsons(self, verbose = False):
        """Load geojsons provided in yaml file and adds the data (GeoDataFrame) as attributes keyed by their key in the yaml. 
        i.e. watershed : "file_name" will read in file_name and can be accessed as self.watershed """
        geo_data = {}
        for id, path in self.geojson.items():
            geodf = load_geojson(path, crs = self.crs)
            geo_data[id] = geodf
            print_verbose(f'loaded {id} geojson', verbose)
        self.__dict__.update(**geo_data, verbose = False)
    def load_csvs(self):
        csv_data = {}
        for id, path in self.csv.items():
            csv = pd.read_csv(path)
            csv_data[id] = csv
            print_verbose(f'loaded {id} csv', verbose)
        self.__dict__.update(**csv_data)
    
    def add_csv_data(self, **path_info):
        csv_data = {}
        for id, path in path_info.items(): 
            csv = pd.read_csv(path)
            csv_data[id] = csv
        self.__dict__.update(**csv_data)



    def load_polygon(self, verbose = False, *geodata_id):
        """Load polygons given a geodata attr. Returns as an attribute to self. 
        i.e. self.load_polygon('watershed') will take self.watershed and extract the polygon data from it.
         polygon data can be accessed using self.watershed_polygon """
        for id in geodata_id:
            geodf = getattr(self, id)
            polygon = extract_polygon(geodf)
            setattr(self, f'{id}_polygon', polygon)
            print_verbose(f"extracted {id} polygon", verbose)

    def load_external_polygon(self, polygon_id : str, polygon_data, verbose = False):
        """Take external polygon GeoDF and adds to self. keyed as {polygon_id}_polygon """
        setattr(self, f'{polygon_id}_polygon', polygon_data)
        print_verbose(f'added {polygon_id}', verbose)

    def merge_polygons(self, merged_polygon_id, verbose = False, *polygons):
        """Takes existing polygons via ID and adds a merged polygon to config keyed under id.
        E.g. self.merge_polygons('merged', 'watershed', 'springshed') will merge the watershed and springshed
        and key it under merged_polygon"""
        polygon_data = [getattr(self, name) for name in polygons]
        merged_polygon = combine_polygons(*polygon_data)
        setattr(self, f'{merged_polygon_id}_polygon', merged_polygon)
        print_verbose(f'merged {polygons}', verbose)

    def set_domain(self, polygon_id : str,  verbose = False): 
        """Takes polygon and returns gdf """
        polygon =  getattr(self, polygon_id)
        self.domain = gpd.GeoDataFrame(geometry = [polygon], crs = self.crs)
        self.total_bounds = self.domain.total_bounds
        print_verbose(f'set domain to {polygon_id}', verbose)

    def apply_DEM_to_domain(self, verbose = False):
        try:
            self.dem_grid = apply_DEM_to_polygon(self.DEM, self.domain, self.delr, self.delc)
            print_verbose(f'applied DEM {self.DEM} to domain',  verbose)
        except(FileNotFoundError):
            print('error, file not found')

    def load_karst_features(self, verbose = False):
        try:
            features = self.karst_features
            for feature_type in features['Type'].unique():
                setattr(self, feature_type, features[features['Type']==feature_type])
                print_verbose(f'added {feature_type}', verbose)
        except(KeyError) as e:
            print(e)        
    def load_creeks(self, verbose = False):
        if isinstance(self.creeks, Creek):
            print('you already have loaded creeks')
        elif isinstance(self.creeks, gpd.GeoDataFrame):
            self.creeks = Creek(geodf = self.creeks)
            print_verbose('loaded new Creek object from creeks data', verbose)
        else:
            print('error')
    def plot_polygons(self, *polygon_names, ax = None, boundary = False, **params):
        if ax == None:
            fig, ax = plt.subplots()
        polygons = [getattr(self, p) for p in polygon_names]
        if boundary == True:
            gpd.GeoDataFrame(geometry = polygons, crs = self.crs).boundary.plot(ax = ax, **params)
        else:
            gpd.GeoDataFrame(geometry = polygons, crs = self.crs).plot(ax = ax, **params)
    def extract_grid_params_from_domain(self, verbose = False): 
        """creates structured grid parameters based """
        self.Lx = self.total_bounds[2] - self.total_bounds[0]
        self.Ly = self.total_bounds[3] - self.total_bounds[1]
        self.ncol = int(self.Lx/self.delc)
        self.nrow = int(self.Ly/self.delr)
        self.nlay = len(self.botm)
        print_verbose(f"""extracted grid params: \nLx = {self.Lx}\nLy = {self.Ly}\nnrow = {self.nrow}\nncol = {self.ncol}\nnlay = {self.nlay}""", verbose)
    
    def extract_top_config(self,  verbose = False):
        if self.top_config == 'DEM':
            self.top = self.dem_grid
            print_verbose('set top to DEM', verbose)
        elif self.top_config == 'constant':
            self.top = self.top_elev
            print_verbose(f'set top to {self.top_elev}', verbose)
        #TODO: add other configurations for custom top array reading in data ETC 

    def extract_bottom_config(self, verbose = False):
        botm_elev = np.zeros((self.nlay, self.nrow, self.ncol))
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
                botm_elev[layer['layer']] = layer_array
        self.botm_elev = botm_elev
        print_verbose(f'set bottom array', verbose)

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

    def plot_model_grid_layers(self, var, **params):
        azim = params.get('azim', None)
        cmap = params.get('cmap', 'viridis')
        elev = params.get('elev', None)
        var_name = params.get('var_name', "")
        show_surface = params.get('show_surface', True)
        vmin = params.get('vmin', None)
        vmax = params.get('vmax', None)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        x, y = np.meshgrid(np.arange(self.ncol) * self.delc + self.total_bounds[0], np.arange(self.nrow)* (-1 * self.delr) + self.total_bounds[3])
        z = np.copy(self.botm_elev)
        if hasattr(self, 'idomain'):
            x[self.idomain[0] == 0] = np.nan
            y[self.idomain[0] == 0] = np.nan
            z[self.idomain == 0] = np.nan
        norm = plt.Normalize(vmin=np.nanmin(var) if vmin is None else vmin, vmax=np.nanmax(var) if vmax is None else vmax)
        cmap_func = plt.cm.get_cmap(cmap)
        for i in range(self.nlay):
            color_layer = cmap_func(norm(var[i]))  # shape (nrow, ncol, 4)
            ax.plot_surface(x, y, z[i], facecolors=color_layer, cmap = cmap)
        if show_surface:
            ax.plot_surface(x, y, self.dem_grid)
        ax.view_init(azim = azim, elev = elev)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_func)
        mappable.set_array([])  # Dummy array needed
        fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label=var_name)
    def extract_idomain(self):
        #set idomain array
        xc_flat = self.model_grid.xcellcenters.ravel()
        yc_flat = self.model_grid.ycellcenters.ravel()
        points = np.column_stack((xc_flat, yc_flat))
        domain = self.domain.values[0]
        shp.prepare(domain)
        
        mask = shp.contains_xy(domain, points[:, 0], points[:, 1])
        idomain_mask = mask.reshape((self.nrow, self.ncol))
        idomain = np.zeros((self.nlay, self.nrow, self.ncol), dtype=int)
        idomain[:, :, :] = idomain_mask.astype(int)
        self.idomain = idomain
    
    def export_idomain(self, path= None, verbose = False):
        if path is None:
            path = self.npy['idomain']
        np.save(path, self.idomain)
        print_verbose(f'saved idomain to {path}', verbose)

    def import_idomain(self, path = None, verbose = False): 
        if path is None:
            path = self.npy['idomain']
        self.idomain = np.load(path)
        print_verbose(f'loaded idomain from {path}', verbose)


    def update_idomain(self, layer, indices, update_val, verbose = False):
        row = indices[:,0]
        col = indices[:,1]
        self.idomain[layer][row, col] = update_val
        print_verbose(f"updated idomain layer {layer} to {update_val}", verbose)

    def extract_K_values(self, verbose = False):
        Kh_array = []
        Kv_array = []
        for K in self.Kh:
            Kh_array.append(np.ones((self.nrow, self.ncol)) * K)
        self.Kh_vals = np.stack(Kh_array, axis = 0)
        for K in self.Kv:
            Kv_array.append(np.ones((self.nrow, self.ncol)) * K)
        self.Kv_vals = np.stack(Kv_array, axis = 0)
        print_verbose('extracted K values', verbose)
    def set_K_values(self, cell_idx : list, Kh = None, Kv = None, verbose = False):
        idx = np.array(cell_idx, dtype=int).T  # shape (3, N)
        layers, rows, cols = idx
        vals = ""
        if Kh is not None:
            self.Kh_vals[layers, rows, cols] = Kh
            vals += f"Kh = {Kh}"
        if Kv is not None:
            self.Kv_vals[layers, rows, cols] = Kv
            vals += f"Kv = {Kv}"
        print_verbose(f'updated {len(cell_idx)} cells with {vals}', verbose)
        #TODO deal with more variable K values
    # def assign_conduit_cells(self):
    #     return
    def extract_creek_cells(self, verbose = False, **params):
        creeks = params.get('creeks', self.creeks)
        self.creek_cells =creeks.map_creek_to_grid(self.nrow, self.ncol, self.delr, self.delc, self.total_bounds[0], self.total_bounds[3])
        self.update_idomain(0, self.creek_cells,1)
        print_verbose('extracted creek cells', verbose)
        return self.creek_cells
    
    def extract_polygon_cells(self, polygon):
        return map_geometry_to_grid([polygon], self.nrow, self.ncol, self.delr, self.delc, self.total_bounds[0], self.total_bounds[3])
    
    def get_cell_elev(self, x, y, is_coordinate = False, **params):
        #given cell index or coordinate, return associated elevation in the elevation array. can handle single point (int or float coordinates) or multiple points (x : array, y : array)
        top = params.get('top', self.top)
        if is_coordinate:
            i, j = self.get_cell_id_from_coords(x, y)
        else:
            i, j = x, y
        return top[i, j]
    
    def get_cell_id_from_coords(self, UTME, UTMN):
        return get_cell_id_from_coords(UTME, UTMN, self.total_bounds[0], self.total_bounds[3], self.delc, self.delr)
    
    def extract_drain_spd(self, verbose = False, **params):
        drain_data = params.get('drain_data', self.drain_data)
        drain_spd = []
        for obj in drain_data:
            if obj['name'] == 'creek':
                C = obj['C']
                row = self.creek_cells[:,0]
                col = self.creek_cells[:,1]
                lay = 0
                elev = self.get_cell_elev(row, col)
                temp_spd = [(lay, r, c, e, C) for r,c,e in zip(row, col, elev)]
                drain_spd += temp_spd
                print_verbose('added creeks to drain spd', verbose)
            elif obj['name'] == 'spring':
                C = obj['C']
                lay = 0
                for idx, spring in self.spring.iterrows():
                    utme = spring.UTME
                    utmn = spring.UTMN
                    row, col = self.get_cell_id_from_coords(utme, utmn)
                    elev = self.get_cell_elev(utme, utmn, True)
                    drain_spd += [(lay, row, col, elev, C)]
                print_verbose('added springs to drain spd', verbose)
            else: 
                print(f"do not recognize {obj['name']}")
        self.drain_spd = drain_spd
        return drain_spd
    def add_drains_module(self, verbose = False, **params):
        #each drain object needs to have cellID (layer, row, and col), elev, and conductance in format (lay, row, col, elev, C)
        gwf = params.get('gwf', self.gwf)
        drn_spd = params.get('drn_spd', self.drain_spd)
        drn = mf.mf6.ModflowGwfdrn(
            gwf,
            stress_period_data = drn_spd,
            save_flows = True,
            print_flows = True
        )
        self.drains = drn
        print_verbose('drains added', verbose)
        return drn
    def set_water_table(self, dtw, verbose = False, **params):
        top = params.get('top', self.top)
        nlay = params.get('nlay', self.nlay)
        idomain = params.get('idomain', self.idomain)
        water_table = top - dtw
        water_table[idomain[0] ==0] = 0
        self.water_table = np.stack([water_table] * nlay, axis = 0)
        print_verbose(f'water table set with dtw = {dtw}', verbose)
        return water_table
    
    def add_npf_module(self, verbose = False, **params): #add node property flow module to sim. assumes that sim has been created
        Kh = params.get('Kh', self.Kh_vals)
        Kv = params.get('Kv', self.Kv_vals)
        icelltype = params.get('icelltype', 0)
        npf = mf.mf6.ModflowGwfnpf(
            self.gwf,
            k = Kh,
            k33 = Kv,
            icelltype = icelltype,
            save_specific_discharge = True
            )
        self.npf = npf 
        print_verbose('npf module created', verbose)

    def extract_recharge(self, **params):
        """takes recharge df and turns into spd compatible with recharge module for MF sim"""
        recharge = params.get('recharge', self.recharge)
        recharge.index = range(len(recharge))
        self.rech = dict(recharge.items())
        return self.rech
    
    def add_recharge_module(self,verbose = False, **params):
        rech = params.get('rech', self.rech)
        rch = mf.mf6.ModflowGwfrcha(self.gwf, recharge = rech) #8.11E-4 Actual recharge value, using 0.00364 m as precip input (daily average for May 2024)
        self.rch = rch
        print_verbose(f'recharge module created with r = {rech}', verbose)

    def add_storage_module(self, verbose = False, **params):
        sy = params.get('sy', self.sy)
        ss = params.get('ss', self.ss)
        sto = mf.mf6.ModflowGwfsto(
            self.gwf,
            sy = sy,
            ss_confined_only = False,
            ss = ss
        )
        self.sto = sto
        print_verbose(f'storage module added with Sy = {sy} and Ss = {ss}', verbose)
    def update_ws(self, ws):
        self.ws = ws
        self.make_sim()
    
    def make_sim(self, verbose = False, **params):
        name = params.get('name', self.name)
        ws = params.get('ws', self.ws) 
        tuni = params.get('tuni', self.tuni)
        nper = params.get('nper', self.nper)
        perlen = params.get('perlen', self.perlen)
        nstp = params.get('nstp', self.nstp)
        tsmult = params.get('tsmult', self.tsmult)
        dtw = params.get('dtw', self.dtw)
        length_unit = 'FEET' if self.lenuni == 1 else 'METERS' if self.lenuni == 2 else 'CENTIMETERS' if self.lenuni == 3 else 'UNKNOWN'
        self.ws = ws
        sim = params.get('sim', mf.mf6.MFSimulation(
            sim_name=name,
            sim_ws=ws,
            exe_name='mf6',
            version = 'mf6'
        ))
        self.sim = sim
        print_verbose("simulation created", verbose)
        tdis = mf.mf6.ModflowTdis(
            sim,
            pname = "tdis",
            time_units = tuni,
            nper = nper,
            perioddata = [(perlen, nstp, tsmult) for _ in range(nper)]
        )
        self.tdis = tdis
        print_verbose(f'time discretization added. {nper} periods with {nstp} {tuni}', verbose)

        ims = mf.mf6.ModflowIms(
            sim,
            pname='ims',
            inner_dvclose = 0.0001,
            outer_dvclose = 0.0001,
            linear_acceleration="BICGSTAB",
            complexity = "SIMPLE",
            print_option = "ALL",
            inner_maximum = 500,
            outer_maximum = 50
        )
        self.ims = ims
        print_verbose('iterative model solver added', verbose)

        #Create the groundwater flow model
        gwf = mf.mf6.ModflowGwf(
            sim,
            modelname = name,
            save_flows = True,
            newtonoptions = 'NEWTON UNDER_RELAXATION'
        )
        self.gwf = gwf
        print_verbose('gwf module added', verbose)

        #create the discretization package
        dis = mf.mf6.ModflowGwfdis(
            gwf,
            nlay = self.nlay,
            nrow = self.nrow,
            ncol = self.ncol,
            delr = self.delr,
            delc = self.delc,
            top = self.top,
            botm = self.botm_elev,
            xorigin = self.total_bounds[0],
            yorigin = self.total_bounds[1],
            idomain = self.idomain,
            length_units = length_unit
        )
        self.dis = dis
        print_verbose('discretization added', verbose)

        self.set_water_table(dtw)
        ic = mf.mf6.ModflowGwfic(gwf, strt = self.water_table)
        self.ic = ic
        print_verbose('set initial conditions', verbose)

        oc = mf.mf6.ModflowGwfoc(gwf,
                            budget_filerecord=f"{name}.bud",
                            head_filerecord=f"{name}.hds",
                            printrecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')],
                            saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
        self.oc= oc
        print_verbose('output control added', verbose)

    def run_sim(self, silent = False):
        self.sim.write_simulation(silent = False)
        success, buff = self.sim.run_simulation(silent = False)
        return success, buff

    def load_sim(self, verbose = False, **params):
        ws = params.get('ws', self.ws)
        name = params.get('name', self.name)
        self.sim = mf.mf6.MFSimulation.load(sim_name=name, sim_ws=ws,exe_name='mf6',
            version = 'mf6')
        print_verbose(f'simulation {ws} loaded', verbose)
    
    def load_model(self, verbose = False, **params):
        name = params.get('name', self.name)
        self.model = self.sim.get_model(name)
        print_verbose(f'model {name} loaded', verbose)
    
    def read_head_output(self, verbose = False, **params):
        totim = params.get('totim', 1)
        try: 
            self.heads = self.model.output.head().get_data(totim = totim)
        except AttributeError:
            self.load_model()
            self.heads = self.model.output.head().get_data(totim = totim)
        self.heads[self.heads > 1e10] = np.nan #clean data by setting very high heads to nan 
        print_verbose('heads data read to heads', verbose)

        return self.heads
    
    def read_drain_discharge_output(self, verbose = False, **params):
        totim = params.get('totim', 1)
        try:
            bud = self.model.output.budget() #read budget
        except AttributeError:
            self.load_model()
            bud = self.model.output.budget() #read budget
        drain_dis = bud.get_data(text='DRN', totim = totim)[0] #get drain discharge from budget
        drain_spd = pd.DataFrame(self.model.get_package('drn').stress_period_data.get_data()[0]) #get drain spd
        dis_arr = np.zeros((self.nlay, self.nrow, self.ncol)) #make discharge array of size nlay x nrow x ncol
        drain_id = [id - 1 for (row, id, dis) in drain_dis] #get cell ids, subtract one to convert to 0-based indexing
        dis = [dis for (row, id, dis) in drain_dis] #get discharge values
        drain_spd.loc[drain_id,'dis'] = dis #append discharge to corresponding drains by ID 
        #populate array with dis values for spatial data 
        for id, q in zip(drain_id, dis):
            loc = drain_spd.loc[id].cellid #get cell index
            dis_arr[loc] = q
        self.drain_spd = drain_spd
        self.drain_array = dis_arr
        if verbose:
            print('read drain discharge to drain_spd and drain_array')
        return drain_spd, dis_arr
    
    def get_timeseries_discharge_single_cell(self, cell_id, **params):
        layer = params.get('layer', 0)
        dis_ts = pd.Series()
        for i in range(self.nper):
            self.read_drain_discharge_output(totim = i+1)
            dis = self.drain_array[layer][cell_id]
            dis_ts.loc[i] = dis
        return dis_ts
    
    def get_timeseries_head_single_cell(self, cell_id, **params):
        layer = params.get('layer', 0)
        dis_ts = pd.Series()
        for i in range(self.nper):
            self.read_drain_discharge_output(totim = i+1)
            dis = self.drain_array[layer][cell_id]
            dis_ts.loc[i] = dis
        return dis_ts
        
    
    def check_head_above_land_surface(self, return_type = 'raw'): 
        if not hasattr(self, 'heads'): #read head output if not done already
            self.read_head_output()
        if return_type == 'raw': #return array of head differences
            return self.heads[0] - self.dem_grid
        elif return_type == 'bool': 
            return self.heads[0] > self.dem_grid
        elif return_type == 'count':
            return np.sum(self.heads[0] > self.dem_grid)
        
    def check_calibration(self, cal_data : CalibrationData):
        return
    


 

    

