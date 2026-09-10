#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
June 11 2026

@author: Jenny Huang
@contact: huan1428@umn.edu

"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

class StorageNode:
    def __init__(self, K, S_y, A, h_0,): 
        """A storage node that has properties kappa (K [L/T]), 
        specific yield (S_y[-]), and surface area (A [L^2]) and initial water height (h_0)"""
        self.K = K
        self.S_y = S_y
        self.h = h_0
        self.A = A

    def update_head(self, R, h_conduit, dt): 
        #compute Q out of node, apply recharge, which is volumetric (?) TODO: finalize implementation-- i think volumetric to be compatible with OK 

        Q = self.K * self.A * (self.h - h_conduit) # if h > h_conduit, then Q > 0 (which is subtracted in the update)
        self.h = self.h + (R - Q) / (self.S_y * self.A) * dt # balance mass, R - Q then / S_y * A 
        return self.h, Q
    

class ConduitNode:
    def __init__(self, h_0, d, L):
        """Simple conduit node that has properties of water height (h [L]), diameter (d [L]), and length (L [L])"""
        self.h = h_0
        self.d = d
        self.L = L
        self.V = np.pi * (d/2)**2 * L
    def update_conduit(self, Q_in, Q_out, dt):
        #update volume based on inflow and outflow, then update head based on new volume 
        self.V = self.V + (Q_in - Q_out) * dt
        self.h = self.V / (np.pi * (self.d/2)**2)
        return self.h, self.V

#example 
K = 1e-5 # m/s
S_y = 0.15 # - 
A = 50e6 # m^2 
h_0_storage = 0 # m

#conduit params
L = 30e3 # m 
d = 1 # m 
h_0_conduit = 0 # m
storage = StorageNode(K = K, S_y = S_y, A = A, h_0 = h_0_storage)
conduit = ConduitNode(h_0 = h_0_conduit, d = d, L = L)
Q_out = 0.5 # m^3/s, constant outflow from conduit for testing

t_max = 1000 # time step in seconds
dt = 1 # time step in seconds
time = np.arange(0, t_max, dt)
h_storage_series = []
h_conduit_series = []
Q_out_series = []
for t in time: 
    R = 0.01 * A # m^3/s, recharge rate (converted to volumetric in update_head)
    h_storage, Q_out = storage.update_head(R, conduit.h, dt)
    h_conduit, V_conduit = conduit.update_conduit(Q_in = Q_out, Q_out = 0.5, dt = dt) # assume constant outflow of 0.5 m^3/s for now
    
    h_storage_series.append(h_storage)
    h_conduit_series.append(h_conduit)
    Q_out_series.append(Q_out)
fig, ax = plt.subplots(2,1, figsize=(10, 5))
ax[0].plot(time,  h_storage_series, label = "Storage")
ax[0].plot(time, h_conduit_series, label = "Conduit")
ax[1].plot(time, Q_out_series, label = "Outflow")
plt.legend()
plt.show()