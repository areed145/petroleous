import colorlover as cl
import plotly.graph_objs as go
import numpy as np
from sklearn import metrics

def serve_kriging_map(mesh,
                      df):

    # Colorscale
    #bright_cscale = [[0, '#FF0000'], [1, '#0000FF']]

    colorscale_zip = zip(np.arange(0, 1.01, 1 / 8),
                          cl.scales['9']['div']['PiYG'])
    cscale = list(map(list, colorscale_zip))

    # Create the plot
    # Plot the prediction contour of the SVM
    trace0 = go.Heatmap(
        x=mesh['x'],
        y=mesh['y'],
        z=mesh['z'],
        hoverinfo='none',
        showscale=False,
        # contours=dict(
        #     showlines=False
        # ),
        colorscale=cscale,
        opacity=0.9
    )

    # Plot Training Data
    trace1 = go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers',
        name=f'Training Data',
        marker=dict(
            size=10,
            color=df['z'],
            colorscale=cscale,
            line=dict(
                width=1
            )
        )
    )

    layout = go.Layout(
        title='Map',
        xaxis=dict(
            # scaleanchor="y",
            # scaleratio=1,
            title='x',
            ticks='',
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title='y',
            ticks='',
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        hovermode='closest',
        legend=dict(x=0, y=-0.01, orientation="h"),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    data = [trace0, trace1]
    figure = go.Figure(data=data, layout=layout)

    return figure

def serve_variogram(svx, svy, svym, sill):

    svs = []
    for i in range(len(svx)):
        svs.append(sill)

    trace0 = go.Scatter(
        x=svx,
        y=svy,
        mode='lines',
        name='Semivariogram',
    )

    trace1 = go.Scatter(
        x=svx,
        y=svym,
        mode='lines',
        name='Model',
    )

    trace2 = go.Scatter(
        x=svx,
        y=svs,
        mode='lines',
        name='Sill',
    )

    layout = go.Layout(
        title='Semivariogram',
        xaxis=dict(
            title='Lag'
        ),
        yaxis=dict(
            title='Semivariance'
        ),
        legend=dict(orientation="h"),
        margin=dict(l=50, r=10, t=55, b=40),
    )

    data = [trace0, trace1, trace2]
    figure = go.Figure(data=data, layout=layout)

    return figure
