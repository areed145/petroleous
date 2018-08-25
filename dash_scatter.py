#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 17:09:16 2018

@author: areed145
"""

import os
import time

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output

import utils.dash_reusable_components as drc
from utils.figures import serve_variogram, serve_kriging_map

import pandas as pd
import random
from pykrige.ok import OrdinaryKriging

app = dash.Dash(__name__)
server = app.server

# Custom Script for Heroku
if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })

    processes = 4

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

    return obs

app.layout = html.Div(children=[
    # .container class is fixed, .container.scalable is scalable
    html.Div(className="banner", style={'backgroundColor': '#00cc99'},
        children=[
        # Change App Name here
        html.Div(className='container scalable', children=[
            # Change App Name here
            html.H2(html.A(
                'Kriging Explorer',
                style={
                    'text-decoration': 'none',
                    'color': 'inherit',
                }
            )),
        ]),
    ]),

    html.Div(id='body', className='container scalable', children=[
        html.Div(className='row', children=[
            html.Div(
                id='div-graphs',
                children=dcc.Graph(
                    id='graph-sklearn-svm',
                    style={'display': 'none'}
                )
            ),

            html.Div(
                className='three columns',
                style={
                    'min-width': '24.5%',
                    'max-height': 'calc(100vh - 85px)',
                    'overflow-y': 'auto',
                    'overflow-x': 'hidden',
                },
                children=[
                    drc.Card([
                        drc.NamedDropdown(
                            name='Select Data',
                            id='dropdown-select-data',
                            options=[
                                {'label': 'Random', 'value': 'rand'},
                                {'label': 'ZoneA', 'value': 'zonea'},
                            ],
                            clearable=False,
                            searchable=False,
                            value='zonea'
                        ),

                        drc.NamedDropdown(
                            name='Select Model',
                            id='dropdown-select-model',
                            options=[
                                {'label': 'Spherical', 'value': 'spherical'},
                                {'label': 'Linear', 'value': 'linear'},
                                {'label': 'Power', 'value': 'power'},
                                {'label': 'Exponential', 'value': 'exponential'},
                                {'label': 'Gaussian', 'value': 'gaussian'},
                                {'label': 'Hole-Effect', 'value': 'hole-effect'},
                            ],
                            clearable=False,
                            searchable=False,
                            value='spherical'
                        ),

                        drc.NamedSlider(
                            name='Sample Size',
                            id='slider-dataset-sample-size',
                            min=100,
                            max=500,
                            step=10,
                            marks={i: i for i in [100, 200, 300, 400, 500]},
                            value=200
                        ),

                        drc.NamedSlider(
                            name='Noise Level',
                            id='slider-dataset-noise-level',
                            min=0,
                            max=2,
                            marks={i: str(i) for i in [0, 0.5, 1, 1.5, 2]},
                            step=0.1,
                            value=0.2,
                        ),

                        drc.NamedSlider(
                            name='Sill',
                            id='slider-dataset-sill',
                            min=0,
                            max=20000,
                            step=1,
                            marks={i: i for i in [0, 4000, 8000, 12000, 16000, 20000]},
                            value=0.778
                        ),

                        drc.NamedSlider(
                            name='Range',
                            id='slider-dataset-rge',
                            min=10,
                            max=10000,
                            step=10,
                            marks={i: i for i in [10,2000,4000,6000,8000,10000]},
                            value=4000
                        ),

                        drc.NamedSlider(
                            name='Nugget',
                            id='slider-dataset-nugget',
                            min=0,
                            max=200,
                            marks={i: i for i in [0,50,100,150,200]},
                            step=5,
                            value=0,
                        ),

                        drc.NamedSlider(
                            name='Neighbors',
                            id='slider-dataset-neighbors',
                            min=5,
                            max=25,
                            marks={i: i for i in [5,10,15,20,25]},
                            step=1,
                            value=6,
                        ),

                        drc.NamedSlider(
                            name='Bins',
                            id='slider-dataset-binct',
                            min=25,
                            max=200,
                            marks={i: i for i in [25,50,100,150,200]},
                            step=25,
                            value=25,
                        ),
                    ]),
                ]
            ),
        ]),
    ])
])

@app.callback(Output('div-graphs', 'children'),
              [Input('dropdown-select-data', 'value'),
               Input('dropdown-select-model', 'value'),
               Input('slider-dataset-sample-size', 'value'),
               Input('slider-dataset-noise-level', 'value'),
               Input('slider-dataset-neighbors', 'value'),
               Input('slider-dataset-rge', 'value'),
               Input('slider-dataset-sill', 'value'),
               Input('slider-dataset-binct', 'value'),
               Input('slider-dataset-nugget', 'value')])
def update_kriging(dat, mod, sample_size, noise, neighbors, rge, sill, binct, nug):
    t_start = time.time()
    lags = 20

    # Data Pre-processing
    if dat == 'rand':
        P = generate_samples(sample_size, noise)
    if dat == 'zonea':
        P = generate_samples(sample_size, 2*noise)

    if mod == 'linear':
        params = {'slope': sill,'nugget': rge}
    elif mod == 'power':
        params = {'scale': sill,'exponent': rge,'nugget': nug}
    else:
        params = {'sill': sill,'range': rge,'nugget': nug}
    
    df = pd.DataFrame(np.array(P), columns=['x','y','z'])
    
    xmin = np.round(P[:,0].min())
    xmax = np.round(P[:,0].max())
    ymin = np.round(P[:,1].min())
    ymax = np.round(P[:,1].max())
            
    mesh = []
    for x in np.arange(xmin, xmax+1).astype(float):
        for y in np.arange(ymin, ymax+1).astype(float):
            mesh.append([x,y])
    mesh = np.array(mesh)

    OK = OrdinaryKriging(P[:,0], P[:,1], P[:,2],
                         variogram_model=mod,
                         verbose=False,
                         variogram_parameters=params,
                         nlags=lags)
    
    z, ss = OK.execute('points', mesh[:,0], mesh[:,1])
    
    mesh = pd.DataFrame(mesh, columns=['x','y'])
    mesh['z'] = z

    prediction_figure = serve_kriging_map(mesh,df)
    
    svx = OK.lags
    svy = OK.semivariance
    svym = OK.variogram_function(OK.variogram_model_parameters, OK.lags)

    variogram_figure = serve_variogram(svx,svy,svym,sill)

    print(
        f"Total Time Taken: {time.time() - t_start:.3f} sec")

    return [
        html.Div(
            className='three columns',
            style={
                'min-width': '24.5%',
                'height': 'calc(100vh - 90px)',
                'margin-top': '5px',

                # Remove possibility to select the text for better UX
                'user-select': 'none',
                '-moz-user-select': 'none',
                '-webkit-user-select': 'none',
                '-ms-user-select': 'none'
            },
            children=[
                dcc.Graph(
                        id='graph-variogram',
                        style={'height': '100%'},
                        figure=variogram_figure
                    ),
                ]),

        html.Div(
            className='six columns',
            style={'margin-top': '5px'},
            children=[
                dcc.Graph(
                    id='graph-sklearn-svm',
                    figure=prediction_figure,
                    style={'height': 'calc(100vh - 90px)'}
                )
            ])
    ]

external_css = [
    # Normalize the CSS
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    # Fonts
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto",
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    # Base Stylesheet, replace this with your own base-styles.css using Rawgit
    "https://rawgit.com/xhlulu/9a6e89f418ee40d02b637a429a876aa9/raw/f3ea10d53e33ece67eb681025cedc83870c9938d/base-styles.css",
    # Custom Stylesheet, replace this with your own custom-styles.css using Rawgit
    "https://cdn.rawgit.com/plotly/dash-svm/bb031580/custom-styles.css"
]

for css in external_css:
    app.css.append_css({"external_url": css})

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)