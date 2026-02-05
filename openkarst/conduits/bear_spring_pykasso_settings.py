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

model_parameters = {

    'outlets' : {
        'number'     : 1,
        'data'       : [[557766.245118, 4869436.0]], #'../../../../data/cave_data/cave_sump.csv',
        'subdomain'  : 'domain_surface',
    },
    'inlets' : {
         'number'     : 5,
        'data'       : '../../../../data/geo_data/sinkholes/sinkholes_dye_trace_2d.txt',#[[558515.0, 4867230.0]], #'../../../../data/geo_data/sinkholes/single_sinkhole_dye_trace.txt',
        'subdomain'  : 'domain_surface',
        'importance' : [1]
    },
    'domain' : {
        'topography' : dem_grid_path,
        'water_level' : np.ones((grid_parameters['nx'], grid_parameters['ny'])) * 375.464,  # flat water table at z=375.464m
    },
    'sks' : {'algorithm' : 'Isotropic3'},
    'fractures' : {'generate': 
                   {'family_01':{ 'density' : 0.00005 , 'orientation' : 135 , 'dip' : 90, 'length' : 300 },  
                    'family_02': { 'density' : 0.00005 , 'orientation' : 45, 'dip' : 90, 'length' : 500 }
                   }
    }

}