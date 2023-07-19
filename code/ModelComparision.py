import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import base64
from VisualAnalysis import FibreGraphs
from Utilities import MachineLearningUtils, DataPreperationUtils
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
import pyDOE
from skimage import io

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

theme =  {
    'dark': True,
    'detail': '#171717',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

encoded_logo = base64.b64encode(open('image_6.png','rb').read())
encoded_totem = base64.b64encode(open('image_5.png','rb').read())

image_fig = go.Figure()

Sigma1=1
Sigma2=1
A=1
BSR=0.01
SPM=200
InputList = []

#empty_graph
empty_layout = go.Layout(
plot_bgcolor='rgba(0,0,0,0)',	
xaxis = dict(showticklabels=False, showgrid=False, zeroline = False),
yaxis = dict(showticklabels = False, showgrid=False, zeroline = False),
height=600, width=800,
)
VisAnaObject = FibreGraphs()
MLUtils = MachineLearningUtils()
DataPrepUtils = DataPreperationUtils()


CVRes_graph_curr = VisAnaObject.getCVtoResolutionGraph([0,0,0,0,0,0,0])
empty_graph = go.Figure()
empty_graph.update_layout(empty_layout)


def getParameterTunerLayout():
    return html.Div([
        html.Div([
            html.Div([
                dcc.Graph(
                id="CVRes_graph",
                figure = CVRes_graph_curr,className="row")
                ])
            ], className="six columns"),
        html.Div([
            html.Div(
                [
                    html.Div([
                        daq.Knob(
                            value=1,
                            id="Sigma_1",
                            label="Sigma_1 (mm)",
                            labelPosition="bottom",
                            size=150,
                            max=50,
                            min=1,
                            style={"background": "transparent"},
                            className="four columns",
                        ),
                        daq.Knob(
                            value=1,
                            id="Sigma_2",
                            label="Sigma_2 (mm)",
                            labelPosition="bottom",
                            size=150,
                            scale={"labelInterval": 10},
                            max=50,
                            min=1,
                            className="four columns",
                        ),
                        daq.Knob(
                            value=1,
                            id="A",
                            label="A",
                            labelPosition="bottom",
                            size=150,
                            scale={"labelInterval": 10},
                            max=50,
                            min=1,
                            className="four columns",
                        )
                    ], className="row"),
                    html.Div([
                        dcc.Input(
                        id="Sigma1",
                        type="number",
                        value=1,
                        min=1,
                        max=50,
                        style={"text-align": "center", "font-size": "15px", "color" : "green" , 'background-color': 'black', 'width': '150px', 'height': '50px'},
                        className="my-container four columns offset-by-one",
                        ),
                        html.Div(id='loop_breaker_container', children=[]),
                        dcc.Input(
                        id="Sigma2",
                        type="number",
                        value=1,
                        min=1,
                        max=50,
                        style={"text-align": "center", "font-size": "15px", "color" : "green" , 'background-color': 'black', 'width': '150px', 'height': '50px'},
                        className="my-container four columns offset-by-two",
                        ),
                        html.Div(id='loop_breaker_container_1', children=[]),
                        dcc.Input(
                        id="A_",
                        type="number",
                        value=1,
                        min=1,
                        max=50,
                        style={"text-align": "center", "font-size": "15px", "color" : "green" , 'background-color': 'black', 'width': '150px', 'height': '50px'},
                        className="my-container four columns offset-by-three",
                        ),
                        html.Div(id='loop_breaker_container_2', children=[])
                    ], className="row"),
                    html.Div([
                        daq.Knob(
                            value=0.01,
                            id="BeltSpinRatio",
                            label="BeltSpinRatio",
                            labelPosition="bottom",
                            size=150,
                            scale={"labelInterval": 10},
                            max=0.25,
                            min=0.01,
                            className="six columns",
                        ),
                        daq.Knob(
                            value=200,
                            id="FibresPerMeter",
                            label="FibresPerMeter",
                            labelPosition="bottom",
                            size=150,
                            scale={"labelInterval": 10},
                            max=10000,
                            min=200,
                            className="six columns",
                        ),
                    ], className="row"),
                    html.Div([
                        dcc.Input(
                        id="BSR",
                        type="number",
                        value=0.01,
                        min=0.01,
                        max=0.25,
                        step=0.01,
                        style={"text-align": "center", "font-size": "15px", "color" : "green" , 'background-color': 'black', 'width': '150px', 'height': '50px'},
                        className="six columns",
                        ),
                        dcc.Input(
                        id="FPM",
                        type="number",
                        value=200,
                        min=200,
                        max=10000,
                        style={"text-align": "center", "font-size": "15px", "color" : "green" , 'background-color': 'black', 'width': '150px', 'height': '50px'},
                        className="six columns",
                        )
                    ], className="row"),
                ], className="knobs",
            )], className="six columns")
        ], className="row")

app.layout = html.Div(
	#title and logo
    html.Div(
        id="colors",
        className="right-panel-controls",
        children=daq.DarkThemeProvider(
        theme=theme,
        children=[
            html.Div
            ([
                html.Div(
                    [
                        html.Div(html.Img(src='data:image/png;base64,{}'.format(encoded_logo.decode()), style={'float':'right'})),
                        html.Div(html.Img(src='data:image/png;base64,{}'.format(encoded_totem.decode()), style={'float':'left'}))
                    ], className='row'
                ),
                html.Div(
                    [
                    html.H1(children='Virtual Nonwoven Analytics',style={ 'fontSize': 32},className='nine columns')
                    ], className="row"
                ),
                #tabs
                dcc.Tabs([
                    #tab for parameter tuning and comparision
                    dcc.Tab(label='Fibre Parameter Tuner', children=
                        [
                            getParameterTunerLayout()
                        ])
                    ])
                ])
            ])
        )
    )

@app.callback(dash.dependencies.Output('Sigma1', 'value'),
	[dash.dependencies.Input('Sigma_1','value')])
def update_CVResGraph(value):
    global Sigma1
    Sigma1 = value
    return value

@app.callback(dash.dependencies.Output('loop_breaker_container', 'children'),
	[dash.dependencies.Input('Sigma1','value')])
def update_loop_breaker(value):
    global Sigma1
    Sigma1 = value
    return [html.Div(id='loop_breaker', children=True)]

@app.callback(dash.dependencies.Output('Sigma_1', 'value'),
    [dash.dependencies.Input('loop_breaker', 'children')])
def update_CVResGraph(val):
    global Sigma1
    return Sigma1

@app.callback(dash.dependencies.Output('Sigma2', 'value'),
	[dash.dependencies.Input('Sigma_2','value')])
def update_CVResGraph(value):
    global Sigma2
    Sigma2 = value
    return value

@app.callback(dash.dependencies.Output('loop_breaker_container_1', 'children'),
	[dash.dependencies.Input('Sigma2','value')])
def update_loop_breaker(value):
    global Sigma2
    Sigma2 = value
    return [html.Div(id='loop_breaker_1', children=True)]

@app.callback(dash.dependencies.Output('Sigma_2', 'value'),
    [dash.dependencies.Input('loop_breaker_1', 'children')])
def update_CVResGraph(val):
    global Sigma2
    return Sigma2

@app.callback(dash.dependencies.Output('A_', 'value'),
	[dash.dependencies.Input('A','value')])
def update_CVResGraph(value):
    global A
    A = value
    return value

@app.callback(dash.dependencies.Output('loop_breaker_container_2', 'children'),
	[dash.dependencies.Input('A_','value')])
def update_loop_breaker(value):
    global A
    A = value
    return [html.Div(id='loop_breaker_2', children=True)]

@app.callback(dash.dependencies.Output('A', 'value'),
    [dash.dependencies.Input('loop_breaker_2', 'children')])
def update_CVResGraph(val):
    global A
    return A

@app.callback(dash.dependencies.Output('BSR', 'value'),
	[dash.dependencies.Input('BeltSpinRatio','value')])
def update_CVResGraph(value):
    global BSR
    BSR = value
    return value

@app.callback(dash.dependencies.Output('loop_breaker_container_3', 'children'),
	[dash.dependencies.Input('BSR','value')])
def update_loop_breaker(value):
    global BSR
    BSR = value
    return [html.Div(id='loop_breaker_3', children=True)]

@app.callback(dash.dependencies.Output('BeltSpinRatio', 'value'),
    [dash.dependencies.Input('loop_breaker_3', 'children')])
def update_CVResGraph(val):
    global BSR
    return BSR

@app.callback(dash.dependencies.Output('FPM', 'value'),
	[dash.dependencies.Input('FibresPerMeter','value')])
def update_CVResGraph(value):
    global SPM
    SPM = value
    return value

@app.callback(dash.dependencies.Output('loop_breaker_container_4', 'children'),
	[dash.dependencies.Input('FPM','value')])
def update_loop_breaker(value):
    global SPM
    SPM = value
    return [html.Div(id='loop_breaker_4', children=True)]

@app.callback(dash.dependencies.Output('FibresPerMeter', 'value'),
    [dash.dependencies.Input('loop_breaker_4', 'children')])
def update_CVResGraph(val):
    global SPM
    return SPM


@app.callback(dash.dependencies.Output('CVRes_graph', 'figure'),
    [dash.dependencies.Input('Sigma_1','value'),
     dash.dependencies.Input('Sigma_2','value'),
     dash.dependencies.Input('A','value'),
     dash.dependencies.Input('BeltSpinRatio','value'),
     dash.dependencies.Input('FibresPerMeter','value')])
def updateParamTunerGraph(val1, val2, val3, val4, val5):
    global Sigma1,Sigma2,A,BSR,SPM, CVRes_graph_curr, InputList
    # InputList.append([Sigma1,Sigma2,A,BSR,SPM])

    # CVRes_graph_curr = VisAnaObject.getCVtoResolutionGraphForComparision(MLUtils.getPredictionList(InputList))
    # InputList.clear()
    # print(Sigma1,Sigma2,A,BSR,SPM)
    CVRes_graph_curr = VisAnaObject.getCVtoResolutionComparisionGraph(MLUtils.comparePredictions([Sigma1,Sigma2,A,BSR,SPM]))
    # print(MLUtils.comparePredictions([Sigma1,Sigma2,A,BSR,SPM]))
    return CVRes_graph_curr

if __name__ == '__main__':
	app.run_server(debug=True)