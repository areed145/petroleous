#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 18:38:09 2018

@author: areed145
"""

import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import geoplot
import model
import kriging

r = random.Random()
r.seed = 42

def generate_samples(n):
    obs = []
    for i in range(n):
        x = r.uniform(-300,300)
        y = r.uniform(-100,100)
        k = r.uniform(-100,100)
        z = np.sqrt((x**2+y**2))-k
        obs.append([x,y,z])
        
    return pd.DataFrame(obs, columns=['x','y','z'])

def upscale_coords(df,bin_sz):
    df['x'] = np.round(df['x']/bin_sz,0)*bin_sz
    df['y'] = np.round(df['y']/bin_sz,0)*bin_sz
    
    return df

def krige_auto(df):

    P = np.array(df)
    
    xmin = np.round(P[:,0].min())
    xmax = np.round(P[:,0].max())
    ymin = np.round(P[:,1].min())
    ymax = np.round(P[:,1].max())
    xr = xmax - xmin
    yr = ymax - ymin
    d = np.sqrt(xr**2+yr**2)
    
    tolerance = 5
    lags = np.arange(tolerance,d,tolerance*2)
    sill = np.var(df['z'])
    
    geoplot.semivariogram(P,lags,tolerance)
    
    svm = model.semivariance(model.spherical,(150,sill))
    geoplot.semivariogram(P,lags,tolerance,model=svm)
    
    covfct = model.covariance(model.spherical,(150,sill))
    
    mesh = []
    for i in np.linspace(xmin,xmax,50):
        for j in np.linspace(ymin,ymax,50):
            mesh.append([i,j])
    
    est,kstd = kriging.krige(P,covfct,mesh,'ordinary',N=5)
    
    mesh = pd.DataFrame(mesh, columns=['x','y'])
    mesh['z'] = est
    
    xi = mesh.x.unique()
    yi = mesh.y.unique()
    zi = mesh.pivot(index='y', columns='x', values='z')
    
    plt.figure(figsize=(12,8))
    plt.contourf(xi, yi, zi, 15)
    plt.scatter(df['x'], df['y'], c=df['z'], marker='o', linewidths=1, edgecolors='white')
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    plt.colorbar()
    plt.savefig('kriged.png',dpi=300)
    
df = generate_samples(150)
krige_auto(df)

#df = upscale_coords(df, 25)
#plt.scatter(df['x'], df['y'], c=df['phi'])