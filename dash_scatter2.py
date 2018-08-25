#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 17:05:36 2018

@author: areed145
"""

import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt

import numpy as np
import random

def generate_samples(n_samples, noise):
    r = random.Random()
    obs = []
    for i in range(n_samples):
        r.seed(i)
        x = r.uniform(-100,100)
        y = r.uniform(-100,100)
        k = r.uniform(-100,100)
        z = np.sqrt((x**2+y**2))-(k*noise)
        obs.append([x,y,z])
        
    obs = np.array(obs)
        
    return obs[:,:2], obs[:,2]

X, y = generate_samples(150,2)

gridx = np.arange(-100, 101, 1).astype(float)
gridy = np.arange(-100, 101, 1).astype(float)

# Create the ordinary kriging object. Required inputs are the X-coordinates of
# the data points, the Y-coordinates of the data points, and the Z-values of the
# data points. If no variogram model is specified, defaults to a linear variogram
# model. If no variogram model parameters are specified, then the code automatically
# calculates the parameters by fitting the variogram model to the binned
# experimental semivariogram. The verbose kwarg controls code talk-back, and
# the enable_plotting kwarg controls the display of the semivariogram.
OK = OrdinaryKriging(X[:,0], X[:,1], y,
                     variogram_model='spherical',
                     verbose=False,
                     enable_plotting=True,
                     variogram_parameters={
                             'sill': 16000,
                             'range': 50,
                             'nugget':0
                             },
                     nlags=10)

# Creates the kriged grid and the variance grid. Allows for kriging on a rectangular
# grid of points, on a masked rectangular grid of points, or with arbitrary points.
# (See OrdinaryKriging.__doc__ for more information.)
z, ss = OK.execute('grid', gridx, gridy)

plt.imshow(z)

# Writes the kriged grid to an ASCII grid file.
kt.write_asc_grid(gridx, gridy, z, filename="output.asc")
