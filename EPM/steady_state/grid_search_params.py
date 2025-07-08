import numpy as np
param_space = {
    'Kh_0' : np.arange(2, 10.1, 1),
    'Kh_1' : np.arange(1, 5.1, 1),
    'Kv_0' : np.arange(2, 10.1, 1),
    'Kv_1' : np.arange(1, 5.1, 1),
    'Kh_0_ss' : np.arange(400, 1001, 100),
    'Kv_0_ss' : np.arange(400, 1001, 100),
    'Kh_1_ss' : np.arange(100, 550, 100),
    'Kv_1_ss' : np.arange(100, 550, 100),
    'C_spring' : np.arange(400, 1001, 100),
    'C_creek' : np.arange(2, 10.1, 1),
}

def param_filter(combo):
    return (combo['Kh_1'] < combo['Kh_0'] 
            and combo['Kv_1'] < combo['Kv_0'] 
            and combo['Kh_0_ss']== combo['C_spring'] 
            and combo['Kh_1_ss'] < combo['Kh_0_ss'] 
            and combo['Kv_1_ss'] < combo['Kv_0_ss']
            and combo['C_creek'] == combo['Kh_0'])