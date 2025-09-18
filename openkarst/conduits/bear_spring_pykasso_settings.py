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
    'sks' : {
        # 'seed' : 1111,
        'mode' : 'A'
    },
    'outlets' : {
        'number'     : 1,
        'data'       : '../../../../data/cave_data/cave_sump.csv',
        'subdomain'  : 'domain',
    },
    'inlets' : {
        'number'     : 9,
        'data'       : '../../../../data/geo_data/sinkholes/sinkholes_dye_trace.txt',
        'subdomain'  : 'domain',
        'importance' : [1]
    },
    'domain' : {
        # 'topography' : dem_grid_path
    },
    'sks' : {'algorithm' : 'Riemann3'}
}