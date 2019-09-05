#these should be imported in DepthProcessor.py:
import argparse
import pandas as pd
import chart_studio.plotly as py
import plotly.graph_objs as go
import numpy as np
from PIL import Image

#this should be in createDataSummary I think..
###3D_url = create3DPlot(self.localDepthDirectory + self.totalHeightChange)

parser = argparse.ArgumentParser(usage="creates a 3D visualization of depth data")
parser.add_argument("file", type=str, help='numpy array containing depth data')
args = parser.parse_args()

project=args.file.split("/")[-2]

def create3DPlot(path):

    depthChange = np.load(path)
    depthChange[depthChange!=depthChange] = 0 #interpolate dummy
    depthChange = np.array(Image.fromarray(depthChange).resize((224,224), resample= Image.BILINEAR))
    depthChange *= 10
    depthChange = depthChange.astype('int8') #see if you can use low precision float

    ''' analyzer:
    for line in depthChange:
        print(line)
    '''

    data = [go.Surface(z=depthChange, colorscale='Viridis', cmax=40, cmin=-40)]

    layout = go.Layout(
        width=800,
        height=700,
        autosize=False,
        title=project+' Depth Change',
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)',
                range = [-40, 40]
            ),
            aspectratio = dict( x=1, y=1, z=0.7 ),
            aspectmode = 'manual'
        )
    )

    fig = dict(data=data, layout=layout)

    # IPython notebook
    # py.iplot(fig, filename='pandas-3d-surface', height=700, validate=False)
    
    url = py.plot(fig, filename=project+"_3d_surface_plot")
    
    #This actually returns a IPython.lib.display.IFrame object
    print(url)
    return url

create3DPlot(args.file)
