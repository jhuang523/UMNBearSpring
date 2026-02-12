import pandas as pd
import numpy as np

def epikarst_reservoir(P, ET, t, S_max, S_0, k, eta, dt_unit = 'D'):
    #k = percolation coefficient (1/t)
    #n = fraction of direct recharge
    #S_0 = initial soil moisture content (0-1)
    #S_max = maximum soil moisture storage (L)
    #P = precipitation (L/t)
    #ET = evapotranspiration (L/t)
    """ precip and evap are used to calculate net infiltration. once soil is saturated (S > S_max), then percolation through epikarst reservoir is calculated as k*S. The outflow from epikarst reservoir is then partitioned into quickflow and slowflow using eta. """
    dt_conversion = {'S': 1, 'M': 60, 'H': 3600, 'D': 86400}
    nt = len(t)
    results = pd.DataFrame()
    S = np.zeros(nt)
    R = np.zeros(nt)
    S[0] = S_0
    for ts in range(1, nt):
        try: 
            dt = (t[ts] - t[ts-1]).total_seconds()/dt_conversion[dt_unit]
        except AttributeError: #if passing an array of floats instead of datetimes
            dt = (t[ts] - t[ts-1])/dt_conversion[dt_unit]
        R[ts] = max(0, S[ts-1] - S_max) * k #recharge to aquifer (L / t)
        S[ts] = max(0, S[ts-1] + ((1-eta)*P[ts] - ET[ts] - R[ts]) * dt) #update soil moisture storage (L)

    results ['date'] = t
    results ['S [L]'] = S
    results ['R_o [L/T]'] = R
    results['R_h [L/T]'] = eta * P
    return results

def linear_reservoir_model(R_l, R_h, t, alpha_l, alpha_h, **params):
    """ Recharge is given in volume per time!"""
    nt = len(t)
    dt_unit = params.get('dt_unit', 'D')
    time_conversion = {'S': 1, 'M': 60, 'H': 3600, 'D': 86400}
    V_l = np.zeros(nt)
    V_l[0] = params.get('V_l0', 0.0)
    V_h = np.zeros(nt)
    V_h[0] = params.get('V_h0', 0.0)
    Q_l = np.zeros(nt)
    Q_h = np.zeros(nt)
    for i in range(1, nt):
        R_l_t = R_l[i]
        R_h_t = R_h[i]
        try: 
            dt = (t[i] - t[i-1]).total_seconds()/time_conversion[dt_unit]
        except AttributeError: #if passing an array of floats instead of datetimes
            print(f"No time objects given, make sure t is in correct units: {dt_unit}")
            dt = (t[i] - t[i-1])
        Q_l[i] = alpha_l * V_l[i-1]
        Q_h[i] = alpha_h * V_h[i-1]   
        V_l[i] = V_l[i - 1] + (R_l_t - Q_l[i]) * dt
    
        V_h[i] = V_h[i - 1] + (R_h_t + Q_l[i]- Q_h[i]) * dt



    results = pd.DataFrame({'date': t, 'R_l [V/T]': R_l, 'R_h [V/T]': R_h, 'V_l [V]': V_l, 'V_h [V]': V_h, 'Q_l [V/T]': Q_l, 'Q_h [V/T]': Q_h})
    return results


def full_model(P, ET, t, A, **params):
    eta = params.get('eta', 0.1)
    S_max = params.get('S_max', 0.3)
    S_0 = params.get('S_0', 0.2)
    k = params.get('k', 0.1)
    alpha_l = params.get('alpha_l', 0.1)
    alpha_h = params.get('alpha_h', 0.01)
    dt_unit = params.get('dt_unit', 'D')
    V_l0 = params.get('V_l0', 0.0)
    V_h0 = params.get('V_h0', 0.0)
    print (f"time unit is {dt_unit}, check your rate parameters!")

    epikarst_reservoir_results = epikarst_reservoir(P, ET, t, S_max = S_max, S_0 = S_0, k = k, eta = eta, dt_unit = dt_unit)
    R_l = epikarst_reservoir_results['R_o [L/T]'] * A
    R_h = epikarst_reservoir_results['R_h [L/T]'] * A
    linear_results = linear_reservoir_model(R_l, R_h, t, alpha_l, alpha_h, dt_unit=dt_unit, V_l0=V_l0, V_h0=V_h0)
    results = pd.concat([epikarst_reservoir_results, linear_results.drop(columns=['date'])], axis=1)
    return results