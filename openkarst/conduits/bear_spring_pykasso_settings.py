import numpy as np
"""Parameters for pykasso bear spring network generation. Import this module to access """
name = 'simple_test'
x0 = 557546
x1 = 560487
y0 = 4867231
y1 = 4869606
z0 = 370
z1 = 400
dx = 5
dy = 5
dz = 5
grid_parameters = {
    'x0': x0,
    'y0': y0,
    'z0' : z0,
    'dx': dx,
    'dy': dy,
    'dz' : dz,
    'nx' : (x1 - x0) // dx,
    'ny' : (y1 - y0) // dy,
    'nz' : (z1 - z0) // dz
}

dem_grid_path = '../../../../data/DEM/dem_grid_bear_spring.npy'
geology = {
    'data'
}
model_parameters = {

    'single_inlet_tortuous' : 
        {'outlets' : {
            'number'     : 1,
            'data'       : [[557766.245118, 4869436.0]], #'../../../../data/cave_data/cave_sump.csv',
            'subdomain'  : 'domain_surface',
        },
        'inlets' : {
            'number'     : 1,
            'data'       : '../../../../data/geo_data/sinkholes/single_sinkhole_dye_trace_2D.txt',#[[558515.0, 4867230.0]], #'../../../../data/geo_data/sinkholes/single_sinkhole_dye_trace.txt',
            'subdomain'  : 'domain_surface',
            'importance' : [1]
        },
        'domain' : {
            'topography' : dem_grid_path,
            'water_level' : np.ones((grid_parameters['ny'], grid_parameters['nx'])) * 375.464,  # flat water table at z=375.464m
        },
        'sks' : {'algorithm' : 'Riemann3'},
        'fractures' : {'generate': 
                    {'family_01':{ 'density' : 0.00005 , 'orientation' : 135 , 'dip' : 90, 'length' : 300 },  
                        'family_02': { 'density' : 0.00005 , 'orientation' : 45, 'dip' : 90, 'length' : 500 }
                    }
        } 
        },



        'dye_trace' : 
        {'outlets' : {
            'number'     : 1,
            'data'       : [[557689.0, 4869553.0]], #'../../../../data/cave_data/cave_sump.csv',
            'subdomain'  : 'domain',
        },
        'inlets' : {
            'number'     : 8,
            'data'       : '../../../../data/geo_data/sinkholes/sinkholes_dye_trace_2D.txt',#[[558515.0, 4867230.0]], #'../../../../data/geo_data/sinkholes/single_sinkhole_dye_trace.txt',
            'subdomain'  : 'domain_surface',
            'importance' : [4, 3, 1]
        },
        'domain' : {
            'water_level' : np.ones((grid_parameters['nx'], grid_parameters['ny'])) *390 ,  # flat water table at z=375.464m
        },
        'sks' : {'algorithm' : 'Riemann3',
                #  'ratio' : 2,
                'karst' : 0.01,
                'geology' : 0.5,
                'fracture' : 0.1,
                'ratio' : 0.1},
        # 'fractures' : {'generate': 
        #             # 'family_01':{ 'density' : 0.0005 , 'orientation' : 135 , 'dip' : 10, 'length' : 300, 'cost' : 0.1 },  
        #             #     'family_02': { 'density' : 0.0005 , 'orientation' : 45, 'dip' : 10, 'length' : 500, 'cost' : 0.1},
        #                 {'family_03':{ 'density' : 0.0005 , 'orientation' : 135 , 'dip' : 90, 'length' : 300, 'cost' : 0.1 },  
        #                 'family_04': { 'density' : 0.0005 , 'orientation' : 45, 'dip' : 90, 'length' : 500, 'cost' : 0.1}
        #             }
        
        # }
        },
        'double_sinkhole' : 
        {'outlets' : {
            'number'     : 1,
            'data'       : [[557689.0, 4869553.0]], #'../../../../data/cave_data/cave_sump.csv',
            'subdomain'  : 'domain_surface',
        },
        'inlets' : {
            'number'     : 6,
            'data'       :    [[560322,4868051], [560322,4868051], [560322,4868051],[560322,4868051], [560322,4868051], [560322,4868051] ],#'../../../../data/geo_data/sinkholes/single_sinkhole_dye_trace.txt',
            'subdomain'  : 'domain_surface',
            'importance' : [1, 1, 1, 1, 1, 1]
        },
        'domain' : {
            'water_level' : np.ones((grid_parameters['ny'], grid_parameters['nx'])) * 390,  # flat water table at z=375.464m
        },
        'sks' : {'algorithm' : 'Riemann3',
                #  'ratio' : 2,
                'karst' : 0.7,
                'geology' : 0.01,
                'fracture' : 0.1,
                'ratio' : 0.1},
        'fractures' : {'generate': 
                    {'family_01':{ 'density' : 0.0005 , 'orientation' : 135 , 'dip' : 90, 'length' : 300, 'cost' : 0.1 },  
                        'family_02': { 'density' : 0.0005 , 'orientation' : 45, 'dip' : 90, 'length' : 500, 'cost' : 0.1}
                    }
        
        }
        },
        'all_sinkholes' : 
        {'outlets' : {
            'number'     : 1,
            'data'       : [[557689.0, 4869553.0]], #'../../../../data/cave_data/cave_sump.csv',
            'subdomain'  : 'domain_surface',
        },
        'inlets' : {
            'number'     : 84,
            'data'       : '../../../../data/geo_data/sinkholes/sinkholes_dye_trace_2D.txt',#[[558515.0, 4867230.0]], #'../../../../data/geo_data/sinkholes/single_sinkhole_dye_trace.txt',
            'subdomain'  : 'domain_surface',
            'importance' : (7 * np.ones(12)).tolist()
        },
        'domain' : {
            'water_level' : np.ones((grid_parameters['ny'], grid_parameters['nx'])) * 375.464,  # flat water table at z=375.464m
        },
        'sks' : {'algorithm' : 'Riemann3',
                #  'ratio' : 2,
                'karst' : 0.01,
                'geology' : 0.5,
                'fracture' : 0.1,
                'ratio' : 0.1},
        'fractures' : {'generate': 
                    {'family_01':{ 'density' : 0.0005 , 'orientation' : 135 , 'dip' : 90, 'length' : 300, 'cost' : 0.1 },  
                        'family_02': { 'density' : 0.0005 , 'orientation' : 45, 'dip' : 90, 'length' : 500, 'cost' : 0.1}
                    }
        
        }
        },
        '5_inlet' : 
        {'outlets' : {
            'number'     : 1,
            'data'       : [[557766.245118, 4869436.0]], #'../../../../data/cave_data/cave_sump.csv',
            'subdomain'  : 'domain_surface',
        },
        'inlets' : {
            'number'     : 40,
            # 'data'       : '../../../../data/geo_data/sinkholes/single_sinkhole_dye_trace_2D.txt',#[[558515.0, 4867230.0]], #'../../../../data/geo_data/sinkholes/single_sinkhole_dye_trace.txt',
            'subdomain'  : 'domain_surface',
            'importance' : (1* np.ones(40)).tolist()
        },
        'domain' : {
            'water_level' : np.ones((grid_parameters['ny'], grid_parameters['nx'])) * 375.464,  # flat water table at z=375.464m
        },
        'sks' : {'algorithm' : 'Riemann3',
                #  'ratio' : 2,
                'karst' : 0.01,
                'geology' : 0.8,
                'fracture' : 0.1,
                'ratio' : 0.1},
        'fractures' : {'generate': 
                    {'family_01':{ 'density' : 0.0005 , 'orientation' : 135 , 'dip' : 90, 'length' : 300, 'cost' : 0.1 },  
                        'family_02': { 'density' : 0.0005 , 'orientation' : 45, 'dip' : 90, 'length' : 500, 'cost' : 0.1}
                    }
        
        }
        },
            'single_inlet_anastomotic' : 
        {'outlets' : {
            'number'     : 1,
            'data'       : [[557766.245118, 4869436.0]], #'../../../../data/cave_data/cave_sump.csv',
            'subdomain'  : 'domain_surface',
        },
        'inlets' : {
            'number'     : 1,
            'data'       : '../../../../data/geo_data/sinkholes/single_sinkhole_dye_trace_2D.txt',#[[558515.0, 4867230.0]], #'../../../../data/geo_data/sinkholes/single_sinkhole_dye_trace.txt',
            'subdomain'  : 'domain_surface',
            'importance' : [1]
        },
        'domain' : {
            'topography' : np.ones((grid_parameters['ny'], grid_parameters['nx'])) * 380,
            'water_level' : np.ones((grid_parameters['ny'], grid_parameters['nx'])) * 375.464,  # flat water table at z=375.464m
        },
        'sks' : {'algorithm' : 'Riemann3'},
        'fractures' : {'generate': 
                    {'family_01':{ 'density' : 0.0005 , 'orientation' : 135 , 'dip' : 90, 'length' : 300 },  
                        'family_02': { 'density' : 0.0005 , 'orientation' : 45, 'dip' : 90, 'length' : 500 }
                    }
        } 
        },
        '5in_5out' : 
        {'outlets' : {
            'number'     : 5,
            'data'       : [[557766.245118, 4869436.0]], #'../../../../data/cave_data/cave_sump.csv',
            'subdomain'  : 'domain_surface',
        },
        'inlets' : {
            'number'     : 5,
            'data'       : '../../../../data/geo_data/sinkholes/single_sinkhole_dye_trace_2D.txt',#[[558515.0, 4867230.0]], #'../../../../data/geo_data/sinkholes/single_sinkhole_dye_trace.txt',
            'subdomain'  : 'domain_surface',
            'importance' : [1, 1, 1, 1,1]
        },
        'domain' : {
            'topography' : np.ones((grid_parameters['ny'], grid_parameters['nx'])) * 380,
            'water_level' : np.ones((grid_parameters['ny'], grid_parameters['nx'])) * 375.464,  # flat water table at z=375.464m
        },
        'sks' : {'algorithm' : 'Riemann3'},
        'fractures' : {'generate': 
                    {'family_01':{ 'density' : 0.0005 , 'orientation' : 135 , 'dip' : 90, 'length' : 300 },  
                        'family_02': { 'density' : 0.0005 , 'orientation' : 45, 'dip' : 90, 'length' : 500 }
                    }
        } 
        },
}