import matplotlib.pyplot as plt

#import plotly
#plotly.tools.set_credentials_file(username='areeder145', api_key='zihvnhFU3CgBqN8RDfcN')

from plotly.offline import download_plotlyjs, init_notebook_mode, plot
import plotly.graph_objs as go

import numpy as np
import random

class well:
    
    def __init__(self, *args, **kwargs):

        self.header = {}
        self.header['well_name'] = kwargs.get('well_name', None)
        self.header['well_type'] = kwargs.get('well_type', None)
        self.header['surf_x'] = kwargs.get('surf_x', None)
        self.header['surf_y'] = kwargs.get('surf_y', None)
        
        self.survey = {}
        
        self.open_hole = {}
        
        self.cased_hole = {}
        
        self.production = {}
        
        self.injection = {}
        
        self.pressure = {}
        
        self.lift = {}
        
class field:
    
    def get_range(self,vmin,vmax,vgrid):
        return np.linspace(vmin,vmax, int((vmax-vmin)/vgrid)+1,endpoint=True)
    
    def plot_contour(self,z,prop,level):
        xv = np.array(self.geology['grid']['x'])
        yv = np.array(self.geology['grid']['y'])
        zv = np.array(self.geology['grid'][z])
        propv = np.array(self.geology['grid'][prop])
        
        xvc = xv[(zv >= level) & (zv < level+self.header['z_grid'])]
        yvc = yv[(zv >= level) & (zv < level+self.header['z_grid'])]
        propvc = propv[(zv >= level) & (zv < level+self.header['z_grid'])]

        contour = go.Contour(
            x=xvc,
            y=yvc,
            z=propvc,
            colorscale='Viridis'
            )
        
        obs = go.Scatter(
            x = self.wells['obs']['surf_x'],
            y = self.wells['obs']['surf_y'],
            marker={'color': 'purple', 
                    'symbol': 17, 
                    'size': 10},
            mode='markers',
            name='obs')
        
        inj = go.Scatter(
            x = self.wells['inj']['surf_x'],
            y = self.wells['inj']['surf_y'],
            marker={'color': 'red', 
                    'symbol': 6, 
                    'size': 10},
            mode='markers',
            name='inj')
        
        prod = go.Scatter(
            x = self.wells['prod']['surf_x'],
            y = self.wells['prod']['surf_y'],
            marker={'color': 'green', 
                    'symbol': 28, 
                    'size': 10},
            mode='markers',
            name='prod')
            
        plot([contour,obs,inj,prod],filename='plot_contour')
    
    def plot_grid(self,z,prop):
        
        grid = go.Scatter3d(
            x=self.geology['grid']['x'],
            y=self.geology['grid']['y'],
            z=self.geology['grid'][z],
            mode='markers',
            marker=dict(
                size=12,
                color=self.geology['grid'][prop],
                colorscale='Viridis',
                #opacity=0.8
            )
        )
            
        plot([grid],filename='plot_grid')
    
    def plot_typelog(self):
        gr = go.Scatter(x=self.geology['typelog']['z'],
                            y=self.geology['typelog']['gr'])
        
        res = go.Scatter(x=self.geology['typelog']['z'],
                            y=self.geology['typelog']['res'])
            
        plot([gr, res],filename='plot_typelog')
    
    def generate_typelog(self,vmin,vmax):
        typelog = []
        for i in self.geology['typelog']['z']:
            typelog.append(self.r.uniform(vmin,vmax))
        return typelog
        
    def generate_geology(self):
        
        self.geology = {}
        
        self.geology['typelog'] = {}
        self.geology['typelog']['z'] = self.get_range(0,self.header['z_max_tl'],self.header['z_grid'])
        self.geology['typelog']['gr'] = self.generate_typelog(0,50)
        self.geology['typelog']['res'] = self.generate_typelog(0,5)
        
        self.geology['grid'] = {}
        self.geology['grid']['x'] = []
        self.geology['grid']['y'] = []
        self.geology['grid']['z'] = []
        self.geology['grid']['tvd'] = []
        self.geology['grid']['gr'] = []
        self.geology['grid']['res'] = []
        
        for x in self.get_range(self.header['x_min'],self.header['x_max'],self.header['x_grid']):
            for y in self.get_range(self.header['y_min'],self.header['y_max'],self.header['y_grid']):
                gl_adj = self.r.uniform(self.header['gl_min'],self.header['gl_max'])
                for idx,z in enumerate(self.geology['typelog']['z']):
                    self.geology['grid']['x'].append(x)
                    self.geology['grid']['y'].append(y)
                    self.geology['grid']['z'].append(z)
                    self.geology['grid']['tvd'].append(z+gl_adj)
                    self.geology['grid']['gr'].append(self.geology['typelog']['gr'][idx]*self.r.uniform(0.8,1.2))
                    self.geology['grid']['res'].append(self.geology['typelog']['res'][idx]*self.r.uniform(0.8,1.2))
        
    def drill_wells(self):
        
        self.wells = {}
        self.wells['welllist'] = {}
        self.wells['prod'] = {}
        self.wells['inj'] = {}
        self.wells['obs'] = {}
        
        for well_type in ['prod','inj','obs']:
            self.wells[well_type]['well_name'] = []
            self.wells[well_type]['surf_x'] = []
            self.wells[well_type]['surf_y'] = []

        for idx in range(self.header['well_ct']):
            well_type = self.r.choice(['prod','inj','obs'])
            well_name = well_type+'_'+str(idx)
            surf_x = self.r.uniform(self.header['x_min'],self.header['x_max'])
            surf_y = self.r.uniform(self.header['y_min'],self.header['y_max'])
            
            self.wells[well_type]['well_name'].append(well_name)
            self.wells[well_type]['surf_x'].append(surf_x)
            self.wells[well_type]['surf_y'].append(surf_y)
            
            self.wells['welllist'][well_name] = well(well_name=well_name, 
                                          well_type=well_type, 
                                          surf_x=surf_x, 
                                          surf_y=surf_y)
            
        
    
    def __init__(self, *args, **kwargs):
        
        self.r = random.Random()
        self.r.seed(420)
        
        self.header = {}
        self.header['name'] = kwargs.get('name', 'field')
        self.header['well_ct'] = kwargs.get('well_ct', 100)
        
        self.header['x_min'] = kwargs.get('x_min', -100)
        self.header['x_max'] = kwargs.get('x_max', 100)
        self.header['y_min'] = kwargs.get('y_min', -100)
        self.header['y_max'] = kwargs.get('y_max', 100)
        
        self.header['x_grid'] = kwargs.get('x_grid', 15)
        self.header['y_grid'] = kwargs.get('y_grid', 15)
        self.header['z_grid'] = kwargs.get('z_grid', 10)
        
        self.header['gl_min'] = kwargs.get('gl_min', -50)
        self.header['gl_max'] = kwargs.get('gl_max', 50)
        self.header['z_max_tl'] = kwargs.get('z_max_tl', 1000)
        
        
        self.generate_geology()
        
        self.drill_wells()
        
        self.plot_typelog()
        
        #self.plot_grid('tvd','res')
        #self.plot_grid('tvd','gr')
        self.plot_grid('z','tvd')
        
        self.plot_contour('tvd','gr',100)

a = field()