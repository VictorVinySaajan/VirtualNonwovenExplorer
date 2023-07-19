from scipy.sparse import bsr
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
# from skimage import io

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

Sigma1=0
Sigma2=0
A=0
BSR=0
SPM=0
TuneParams = 1
InputList = []
previous_clicks_add, previous_clicks_remove, previous_clicks_clear, previous_clicks_heatmap= 0,0,0,0
cluster_click1,cluster_click2,cluster_click3,cluster_click4,cluster_click5=0,0,0,0,0
previous_clicks_add_t, previous_clicks_remove_t, previous_clicks_clear_t = 0,0,0
individual_resolutions_on = "Multiple grid-sizes"
baseweight_text = "Switch to BaseWeight Analysis"
p_fpm_val, p_bsr_val = 0,0
bsr_min, bsr_max, spm_min, spm_max= 0.01,0.25,200,10000
prevSampleGenerationClicks = 0
frozen_input = 'None'
fullRangePredictionsGlobal = pd.DataFrame()
trace_count = 0
FocalPoint = 1
FocalPointData = 0
Resolution = 0
DResolution = 0
tick_box_array=[]
curr_number_input_1, curr_number_input_2, curr_number_input_3, curr_number_input_4, curr_number_input_5 = 0,0,0,0,0
prevUserFocalPoint, prevIndex = 0 , 0
titer = 1.1
emw = titer * 200 * 0.01
sigma1_derivative_cur, sigma2_derivative_cur, a_derivative_cur, bsr_derivative_cur, spm_derivative_cur, frozen_deriv_ip = 0,0,0,0,0,"Sigma_1"
subrange_min_1,subrange_max_1,subrange_min_2,subrange_max_2 = 1,50,1,50

theme =  {
    'dark': True,
    'detail': '#171717',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

encoded_logo = base64.b64encode(open('code/image_6.png','rb').read())
encoded_totem = base64.b64encode(open('code/image_5.png','rb').read())

# surro_img = io.imread('../images/sample_1.png')
# image_fig = px.imshow(surro_img)
image_fig = go.Figure()

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


clusterInputData, clusterOutputData, clusterMean, boxplotDataframe = DataPrepUtils.getClusterData()
clusterOp = VisAnaObject.getClusterGraph(clusterOutputData)
clusterOp.update_layout(title="Output Clusters")
clusterIp = VisAnaObject.getClusterGraph(clusterInputData)
clusterIp.update_layout(title="Input Clusters")
currClusterMean = clusterMean[0]
meanPlot =  px.bar(x=["Half","One","Two","Five","Ten","Twenty","Fifty"], y=currClusterMean[:-1], title="Mean CV Value Vs grid-size")
meanPlot.update_yaxes(range=[0, 100])
boxPlot = px.box(y=boxplotDataframe[0].Sigma_1, title="Sigma_1")
boxPlot1 = px.box(y=boxplotDataframe[0].Sigma_2, title="Sigma_2")
boxPlot2 = px.box(y=boxplotDataframe[0].A, title="A")
boxPlot3 = px.box(y=boxplotDataframe[0].BeltSpinRatio, title="BletSpinRatio")
boxPlot4 = px.box(y=boxplotDataframe[0].SpinPositionsPerMeterInverse, title="SpinPositionsPerMeter")

CVRes_graph_curr = VisAnaObject.getCVtoResolutionGraph([0,0,0,0,0,0,0])
cli_graph_curr = VisAnaObject.getCliGraph(0)
empty_graph = go.Figure()
empty_graph.update_layout(empty_layout)
df = pd.DataFrame(columns = ['Trace_Number','Sigma_1', 'Sigma_2', 'A', 'BeltSpinRatio','FibresPerMeter'])
tableList = list()

fullRangePredictions = DataPrepUtils.getFullRangePredictions('S1S2', 1, 0.01, 200,25, Resolution,1,50,1,50)
Surface_graph_global = VisAnaObject.get3DSurfacePlot('Sigma_1 and Sigma_2', fullRangePredictions, Resolution)

def MapToLinSpace(dimension, lowerBound, upperBound):
    return np.multiply(dimension, (upperBound-lowerBound)) + lowerBound

def GetLatinHypercubeSamples(dimensions, sampleSize):
    dataframe = pd.DataFrame(columns = ["Sigma_1", "Sigma_2", "A", "BeltSpinRatio", "SpinPositionsPerMeterInverse"])
    samples = pyDOE.lhs(dimensions, samples=sampleSize)
    samples[:, 0] = MapToLinSpace(samples[:, 0], 1, 50)
    samples[:, 1] = MapToLinSpace(samples[:, 1], 1, 50)
    samples[:, 2] = MapToLinSpace(samples[:, 2], 1, 50)
    samples[:, 3] = MapToLinSpace(samples[:, 3], 0.01, 0.25)
    samples[:, 4] = MapToLinSpace(samples[:, 4], 200, 10000)
    dataframe['Sigma_1'] = samples[:, 0]
    dataframe['Sigma_2'] = samples[:, 1]
    dataframe['A'] = samples[:, 2]
    dataframe['BeltSpinRatio'] = samples[:, 3]
    dataframe['SpinPositionsPerMeterInverse'] = samples[:, 4]
    return dataframe

dframe = GetLatinHypercubeSamples(5, 100)
scaler = preprocessing.StandardScaler().fit(dframe)  
pc = PCA(n_components=2)
pc.fit(scaler.transform(dframe))
z = pc.transform(scaler.transform(dframe))

dframe1 = DataPrepUtils.getOutputSpacePrediction(dframe)
scaler1 = preprocessing.StandardScaler().fit(dframe1)
pc1 = PCA(n_components=2)
pc1.fit(scaler1.transform(dframe1))
z1 = pc1.transform(scaler1.transform(dframe1))

gmm = GaussianMixture(n_components=5)
gmm.fit(z1)

dframe_display = dframe.copy()
dframe_display['pc_1'] = z[:,0]
dframe_display['pc_2'] = z[:, 1]
#Cluster output date and map that to input
dframe_display['ClusterColor'] = gmm.predict(z1)



InputSpaceGraph = px.scatter(dframe_display, x='pc_1', y='pc_2', color = 'ClusterColor',
                            hover_data = {'Sigma_1' : True,
                                          'Sigma_2' : True,
                                          'A' : True,
                                          'BeltSpinRatio' : True,
                                          'SpinPositionsPerMeterInverse' : True,
                                          'pc_1' : False,
                                          'pc_2' : False,
                                          'ClusterColor': False
                                         })

dframe1_display = pd.DataFrame()
dframe1_display['Res: 0.5'] = dframe1[:,0]
dframe1_display['Res: 1'] = dframe1[:,1]
dframe1_display['Res: 2'] = dframe1[:,2]
dframe1_display['Res: 5'] = dframe1[:,3]
dframe1_display['Res: 10'] = dframe1[:,4]
dframe1_display['Res: 20'] = dframe1[:,5]
dframe1_display['Res: 50'] = dframe1[:,6]


dframe1_display['pc_1'] = z1[:, 0]
dframe1_display['pc_2'] = z1[:, 1]
dframe1_display['ClusterColor'] = dframe_display['ClusterColor']
OutputSpaceGraph = px.scatter(dframe1_display, x='pc_1', y='pc_2', color = 'ClusterColor',
                            hover_data = {'Res: 0.5' : True,
                                          'Res: 1' : True,
                                          'Res: 2' : True,
                                          'Res: 5' : True,
                                          'Res: 10' : True,
                                          'Res: 20' : True,
                                          'Res: 50' : True,
                                          'pc_1' : False,
                                          'pc_2' : False,
                                          'ClusterColor': False
                                         })

dframe3 = dframe.copy()
dframe3['0.5'] = dframe1[:,0]
dframe3['1'] = dframe1[:,1]
dframe3['2'] = dframe1[:,2]
dframe3['5'] = dframe1[:,3]
dframe3['10'] = dframe1[:,4]
dframe3['20'] = dframe1[:,5]
dframe3['50'] = dframe1[:,6]
dframe3['pc1_ip'] = z[:, 0]
dframe3['pc2_ip'] = z[:, 1]
dframe3['pc1_op'] = z1[:, 0]
dframe3['pc2_op'] = z1[:, 1]
dframe3['Label_op'] = gmm.predict(z1)

ParallelCoordinatesGraph = go.Figure(data=
    go.Parcoords(
        line_color='grey',
        dimensions = list([
            dict(range = [1,50],
                 label = "Sigma_1", values = dframe3['Sigma_1']),
            dict(range = [1,50],
                 label = 'Sigma_2', values = dframe3['Sigma_2']),
            dict(range = [1,50],
                 label = 'A', values = dframe3['A']),
            dict(range = [0.01,0.25],
                 label = 'BSR', values = dframe3['BeltSpinRatio']),
            dict(range = [200,10000],
                 label = 'SPM', values = dframe3['SpinPositionsPerMeterInverse']),
            dict(
                 label = '0.5*0.5', values = dframe3['0.5']),
            dict(
                 label = '1*1', values = dframe3['1']),
            dict(
                 label = '2*2', values = dframe3['2']),
            dict(
                 label = '5*5', values = dframe3['5']),
            dict(
                 label = '10*10', values = dframe3['10']),
            dict(
                 label = '20*20', values = dframe3['20']),
            dict(
                 label = '50*50', values = dframe3['50'])])
    )
)

derivData = DataPrepUtils.getDerivativeDataFrame("Sigma_1", 1, 1, 0.01, 200, DResolution, 100)
derivPlot = px.scatter()
derivPlot = go.Figure(data=go.Scatter(x=derivData['Sigma_1'], y=derivData['Derivative']))
derivPlot.update_layout(
    title="Partial Derivative Plot",
    xaxis_title="Sigma_1",
    yaxis_title="Derivative",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
# derivPlot = px.scatter(derivData, x="Resolution", y="Derivative", animation_frame="Sigma_1", animation_group="Resolution", 
#             range_y=[derivData['Derivative'].min() - 5, derivData['Derivative'].max() + 5], size = "Size")


def getParameterTunerLayout():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                        dcc.Graph(
                        id="CVRes_graph",
                        figure = CVRes_graph_curr,className="eight columns"),
                        dcc.Graph(
                        id="Cli_graph",
                        figure = cli_graph_curr,className="four columns")
                ]),
                ], className="row"),
                html.Div([
                html.Div(
                children=dash_table.DataTable(
                    id="table",
                    columns=[{"name": i, "id": i} for i in df.columns],
                ), style={'color': 'blue', 'font-size': '15px', 'font-weight': 'bold'}),
                # dcc.Graph(
                # id='image_fig',
                # figure = image_fig, className="six columns")
                ], className = "row")
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
						
						daq.LEDDisplay(
							id="Sigma1",
							size=20,
							value=1,
							label="Sigma_1 (mm)",
							labelPosition="bottom",
							color=theme["primary"],
							className="four columns",
						),
                        html.Div(id='loop_breaker_container', children=[]),
						daq.LEDDisplay(
							id="Sigma2",
							size=20,
							value=1,
							label="Sigma_1 (mm)",
							labelPosition="bottom",
							color=theme["primary"],
							className="four columns",
						),
                        html.Div(id='loop_breaker_container_1', children=[]),
						daq.LEDDisplay(
							id="A_",
							size=20,
							value=1,
							label="Sigma_1 (mm)",
							labelPosition="bottom",
							color=theme["primary"],
							className="four columns",
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
                            max=0.25,
                            min=0.01,
                            className="four columns",
                        ),
                    html.Div([
                    html.Div([
                        html.Div([
                        html.Div(id= "switch",children= "Multiple grid-sizes",style={"text-align": "center", "font-size":"200%", "color":"black"},
                        className="six columns"),
                        html.Div(id= "bwa_",children= "Switch to BaseWeight Analysis",style={"text-align": "center", "font-size":"200%", "color":"black"},
                        className="six columns")
                        ], className="row"),
                        html.Div([
                        daq.PowerButton(
                            id="power",
                            on='True',
                            size=80,
                            className="six columns"
                        ),
                        daq.PowerButton(
                            id="image",
                            on='False',
                            size=80,
                            className="six columns"
                        )], className="row"),
                        html.Pre(children= "Enter BaseWeight in [g/m^2]",style={"text-align": "center", "font-size":"200%", "color":"black"},
                        className="row"),
                        dcc.Input(
                                id="emw",
                                type="number",
                                value=emw,
                                # label="Enter the base weight in [g/m^2]",
                                # labelPosition="top",
                                style={"text-align": "center", "font-size": "15px", "color" : "red" , 'background-color': 'black'},
                                className="row",
                        )
                    ], className = "my_input_box_container",
                    style={'width' : '100%'})
                    ], className="four columns"),
                        daq.Knob(
                            value=200,
                            id="FibresPerMeter",
                            label="FibresPerMeter",
                            labelPosition="bottom",
                            size=150,
                            scale={"labelInterval": 10},
                            max=10000,
                            min=200,
                            className="four columns",
                        ),
                    ], className="row"),
                    html.Div([
                        daq.LEDDisplay(
                            id="BSR",
                            size=20,
                            value=0.01,
                            label="BeltSpinRatio",
                            labelPosition="bottom",
                            color=theme["primary"],
                            className="four columns",
                        ),
                        html.Div(id='loop_breaker_container_3', children=[]),
                        html.Div([
                            html.Pre(children= "titer(dtex)",style={"top":"50%","margin-top":"-1px", "text-align": "center", "font-size":"200%", "color":"black"},
                            className="six columns"),
                        dcc.Input(
                        id="titer",
                        type="number",
                        value=titer,
                        style={"text-align": "center", "font-size": "15px", "color" : "green" , 'background-color': 'black'},
                        className="four columns"
                        )
                        ], className="four columns"),
                        daq.LEDDisplay(
                            id="FPM",
                            size=20,
                            value=200,
                            label="Fibres Per Meter",
                            labelPosition="bottom",
                            color=theme["primary"],
                            className="four  columns",
                        ),
                        html.Div(id='loop_breaker_container_4', children=[])
                    ], className="row"),
                    html.Div([
                        html.Button('ADD INPUT', id='ADD', n_clicks=0, className="four columns"),
                        html.Button('REMOVE INPUT', id='REMOVE', n_clicks=0, className="four columns"),
                        html.Button('CLEAR ALL', id='CLEAR', n_clicks=0, className="four columns"),
                        html.Div(id='hidden-div', style={'display':'none'})
                    ], className="row"),
                ], className="knobs",
            )], className="six columns")
        ], className="row")

def get3DSurfacePlotLayout():
    return html.Div([
        html.Div([
            dcc.Graph(
            id="Surface_graph",
            figure = Surface_graph_global,
        )],className="six columns"),
    html.Div([
        html.Div(
            [
            html.Div([
                daq.Knob(
                    value=1,
                    id="Sigma_11",
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
                    id="Sigma_21",
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
                    id="A1",
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
                daq.LEDDisplay(
                    id="Sigma11",
                    size=20,
                    value=2,
                    label="Sigma_1 (mm)",
                    labelPosition="bottom",
                    color=theme["primary"],
                    className="four columns",
                ),
                daq.LEDDisplay(
                    id="Sigma21",
                    size=20,
                    value=2,
                    label="Sigma_2 (mm)",
                    labelPosition="bottom",
                    color=theme["primary"],
                    className="four columns",
                ),
                daq.LEDDisplay(
                    id="A_1",
                    size=20,
                    value=2,
                    label="A",
                    labelPosition="bottom",
                    color=theme["primary"],
                    className="four columns",
                )
            ], className="row"),
            html.Div([
                daq.Knob(
                    value=0.01,
                    id="BeltSpinRatio1",
                    label="BeltSpinRatio",
                    labelPosition="bottom",
                    size=150,
                    scale={"labelInterval": 10},
                    max=0.25,
                    min=0.01,
                    className="four columns",
                ),
                html.Div([
                    html.Div(id= "subrange",children= "Select Subrange",style={"text-align": "center", "font-size":"200%", "color":"black"}
                    ,className="row"),
                    html.Div([
                        html.Div(id= "option_1",children= "Sigma_1",style={"text-align": "center", "font-size":"200%", "color":"black"},
                        className = "six columns"),
                        html.Div(id= "option_2",children= "Sigma_2",style={"text-align": "center", "font-size":"200%", "color":"black"},
                        className = "six columns")
                    ],className="row"),
                    html.Div([
                        html.Div(id= "option_1_min",children= "Min",style={"text-align": "center", "font-size":"200%", "color":"blue"},
                        className = "three columns"),
                    dcc.Input(id="min_1", type="number", placeholder="1", className = "three columns"),
                    html.Div(id= "option_2_min",children= "Min",style={"text-align": "center", "font-size":"200%", "color":"blue"},
                        className = "three columns"),
                        dcc.Input(id="min_2", type="number", placeholder="1", className = "three columns")
                    ], className="row"),
                    html.Div([
                        html.Div(id= "option_1_max",children= "Max",style={"text-align": "center", "font-size":"200%", "color":"blue"},
                        className = "three columns"),
                    dcc.Input(id="max_1", type="number", placeholder="50", className = "three columns"),
                    html.Div(id= "option_2_max",children= "Max",style={"text-align": "center", "font-size":"200%", "color":"blue"},
                        className = "three columns"),
                        dcc.Input(id="max_2", type="number", placeholder="50", className = "three columns")
                    ], className="row"),
                    html.Div([
                        html.Div(id= "sample_number",children= "Samples",style={"text-align": "center", "font-size":"200%", "color":"black"},
                        className = "six columns"),
                        dcc.Input(id="sam_num", type="number", placeholder="50", className = "six columns")
                    ], className="row")
                ], className="four columns"),
                daq.Knob(
                    value=200,
                    id="FibresPerMeter1",
                    label="FibresPerMeter",
                    labelPosition="bottom",
                    size=150,
                    scale={"labelInterval": 10},
                    max=10000,
                    min=200,
                    className="four columns",
                ),
            ], className="row"),
            html.Div([
                daq.LEDDisplay(
                    id="BSR1",
                    size=20,
                    value=2,
                    label="BeltSpinRatio",
                    labelPosition="bottom",
                    color=theme["primary"],
                    className="six columns",
                ),
                daq.LEDDisplay(
                    id="FPM1",
                    size=20,
                    value=2,
                    label="Fibres Per Meter",
                    labelPosition="bottom",
                    color=theme["primary"],
                    className="six columns",
                )
            ], className="row"),
            html.Div([                     
				html.Div([
					dcc.Dropdown(
    				options=[
        					{'label': 'Sigma_1 vs Sigma_2', 'value': 'S1S2'},
                            {'label': 'Sigma_1 vs A', 'value': 'S1A'},
                            {'label': 'Sigma_1 vs BeltSpinRatio', 'value': 'S1BSR'},
                            {'label': 'Sigma_1 vs FibresPerMeter', 'value': 'S1SPM'},
                            {'label': 'Sigma_2 vs A', 'value': 'S2A'},
                            {'label': 'Sigma_2 vs BeltSpinRatio', 'value': 'S2BSR'},
                            {'label': 'Sigma_2 vs FibresPerMeter', 'value': 'S2SPM'},
                            {'label': 'A vs BeltSpinRatio', 'value': 'ABSR'},
                            {'label': 'A vs FibresPerMeter', 'value': 'ASPM'},
                            {'label': 'BeltSpinRatio vs FibresPerMeter', 'value': 'BSRSPM'}
    					],
                        placeholder="Select the combination of parameters to freeze",
    					id='Inputs'
					)],className = "three columns"),
                html.Div([
					dcc.Dropdown(
    				options=[
        					{'label': '0.5', 'value': 0},
                            {'label': '1', 'value': 1},
                            {'label': '2', 'value': 2},
                            {'label': '5', 'value': 3},
                            {'label': '10','value' : 4},
                            {'label': '20', 'value': 5},
                            {'label': '50', 'value': 6},
                            {'label': 'Cli', 'value': 7}
    					],
                        placeholder="Choose the grid-size or cli",
    					id='resolution'
					)],className = "three columns"),
                    html.Button('3D SURFACE PLOT', id='3DSP', n_clicks=0, className="three columns"),
                    html.Button('HEAT MAP', id='HEAT', n_clicks=0, className="three columns")
            ], className="row"),
        ], className="knobs",
    )], className="six columns")
    ], className="row")

def getDetailOnDemandLayout():
    return  html.Div([
                html.Div([
                html.Div([
                    dcc.Graph(
                    id="InputSpaceGraph",
                    figure = InputSpaceGraph,className="four columns")
                    ]),
                html.Div([
                    dcc.Graph(
                    id="OutputSpaceGraph",
                    figure = OutputSpaceGraph,className="four columns")
                    ]),
                html.Div([
                    html.Div([
                    html.Pre(children= "Enter the Focal Point of Interest",
                    style={"text-align": "center", "font-size":"200%", "color":"black"})
                    ]),
                    html.Div([
                        dcc.Input(
                        id="number_input_1",
                        type="number",
                        placeholder="Sigma_1",
                        className="row"
                        ),
                        dcc.Input(
                        id="number_input_2",
                        type="number",
                        placeholder="Sigma_2",
                        className="row"
                        ),
                        dcc.Input(
                        id="number_input_3",
                        type="number",
                        placeholder="A",
                        className="row"
                        )
                        ]),
                        dcc.Input(
                        id="number_input_4",
                        type="number",
                        placeholder="BeltSpinRatio",
                        className="row"
                        ),
                        dcc.Input(
                        id="number_input_5",
                        type="number",
                        placeholder="SpinPositionsPerMeter",
                        className="row"
                        ),
                        html.Button('Start Analysis', id='FP', n_clicks=0),
                        html.Div([
                            dcc.Checklist(
                            options=[
                            {'label': 'Sigma-1', 'value': 1},
                            {'label': 'Sigma-2', 'value': 2},
                            {'label': 'A', 'value': 3},
                            {'label': 'BeltSpinRatio', 'value': 4},
                            {'label': 'SpinPositionsPerMeter', 'value': 5}
                            ],
                            id = 'tick-box',
                            className='my_box_container',
                            inputClassName='my_box_input', 
                            labelClassName='my_box_label',)
                        ]),
                        html.Button('Generate Samples around Focal Point', id='FP_Samples', n_clicks=0)
                    ], className = "my_input_box_container",
                    style={'width' : '25%'})
                    ], className = "row"),
                html.Div(id="dummy"),
                html.Div([
                html.Div([
                    dcc.Graph(
                    id="ParallelCoordinatesGraph",
                    figure = ParallelCoordinatesGraph)
                    ])
                ], className="row")
            ])

def getClusterAnalysisLayout():
    return  html.Div([
                html.Div([   
                    html.Div([
                    dcc.Graph(
                    id="OutputCluster",
                    figure = clusterOp,className="four columns")
                    ]),
                    html.Div([
                    dcc.Graph(
                    id="InputCluster",
                    figure = clusterIp,className="three columns")
                    ]),
                    html.Div([
                    dcc.Graph(id="MeanPlot", figure =meanPlot,className="four columns")
                    ]),
                    html.Div([
                        html.Div([
                            html.Button('Cluster1', id='Cluster1', n_clicks=0, className="row"),
                            html.Button('Cluster2', id='Cluster2', n_clicks=0, className="row"),
                            html.Button('Cluster3', id='Cluster3', n_clicks=0, className="row"),
                            html.Button('Cluster4', id='Cluster4', n_clicks=0, className="row"),
                            html.Button('Cluster5', id='Cluster5', n_clicks=0, className="row")
                        ])
                    ], className="one columns")
                ], className="row"),
                html.Div([
                    dcc.Graph(id="boxPlot", figure =boxPlot,className="two columns"),
                    dcc.Graph(id="boxPlot1", figure =boxPlot1,className="two columns"),
                    dcc.Graph(id="boxPlot2", figure =boxPlot2,className="two columns"),
                    dcc.Graph(id="boxPlot3", figure =boxPlot3,className="two columns"),
                    dcc.Graph(id="boxPlot4", figure =boxPlot4,className="two columns")
                ], className="row")
            ])

def getDerivativeAnalysisLayout():
    return html.Div([
            html.Div([
                dcc.Graph(
                id="DerivativePlot",
                figure = derivPlot
            )],className="six columns"),
        html.Div([
            html.Div(
                [
                html.Div([
                    daq.Knob(
                        value=1,
                        id="DS1",
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
                        id="DS2",
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
                        id="DA",
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
                    daq.LEDDisplay(
                        id="DLS1",
                        size=20,
                        value=2,
                        label="Sigma_1 (mm)",
                        labelPosition="bottom",
                        color=theme["primary"],
                        className="four columns",
                    ),
                    daq.LEDDisplay(
                        id="DLS2",
                        size=20,
                        value=2,
                        label="Sigma_2 (mm)",
                        labelPosition="bottom",
                        color=theme["primary"],
                        className="four columns",
                    ),
                    daq.LEDDisplay(
                        id="DLA",
                        size=20,
                        value=2,
                        label="A",
                        labelPosition="bottom",
                        color=theme["primary"],
                        className="four columns",
                    )
                ], className="row"),
                html.Div([
                    daq.Knob(
                        value=0.01,
                        id="DBSR",
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
                        id="DSPM",
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
                    daq.LEDDisplay(
                        id="DLBSR",
                        size=20,
                        value=2,
                        label="BeltSpinRatio",
                        labelPosition="bottom",
                        color=theme["primary"],
                        className="six columns",
                    ),
                    daq.LEDDisplay(
                        id="DLSPM",
                        size=20,
                        value=2,
                        label="Fibres Per Meter",
                        labelPosition="bottom",
                        color=theme["primary"],
                        className="six columns",
                    )
                ], className="row"),
                html.Div([                     
					dcc.Dropdown(
    				options=[
        					{'label': 'Sigma_1', 'value': 'DS1'},
                            {'label': 'Sigma_2', 'value': 'DS2'},
                            {'label': 'A', 'value': 'DA'},
                            {'label': 'BeltSpinRatio', 'value': 'DBSR'},
                            {'label': 'FibresPerMeter', 'value': 'DSPM'}
    					],
                        placeholder="Select the derivative parameter",
    					id='DInputs',
                        className = "six columns"
					),
					dcc.Dropdown(
    				options=[
        					{'label': '0.5', 'value': 0},
                            {'label': '1', 'value': 1},
                            {'label': '2', 'value': 2},
                            {'label': '5', 'value': 3},
                            {'label': '10','value' : 4},
                            {'label': '20', 'value': 5},
                            {'label': '50', 'value': 6},
                            {'label': 'All', 'value': 7}
    					],
                        placeholder="Choose the grid-size",
    					id='dresolution',
                        className = "six columns"
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
                        ]),

                    dcc.Tab(label='3D Surface plot', children=
                        [
                            get3DSurfacePlotLayout()
                        ]),
                    dcc.Tab(label='Sensitivity Analysis', children=
                        [
                            getDetailOnDemandLayout()
                        ]),
                    dcc.Tab(label='Cluster Analysis', children=
                        [
                            getClusterAnalysisLayout()
                        ]),
                    dcc.Tab(label='Derivative Analysis', children=
                        [
                            getDerivativeAnalysisLayout()
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

@app.callback([dash.dependencies.Output('BSR', 'value'),
     dash.dependencies.Output('FibresPerMeter','value')],
	[dash.dependencies.Input('BeltSpinRatio','value')])
def update_CVResGraph(value):
    global BSR, SPM, spm_min, spm_max

    if value is not None:
        if baseweight_text == "Switch to Overall Analysis":
            if value < bsr_min:
                SPM = emw/(titer * bsr_min) 
            elif value >= bsr_max:
                SPM = emw/(titer * bsr_max) 
            else:
                SPM = emw/(titer * value)

    BSR = value

    return [BSR, round(SPM,4)]

# @app.callback(dash.dependencies.Output('loop_breaker_container_3', 'children'),
# 	[dash.dependencies.Input('BSR','value')])
# def update_loop_breaker(value):
#     global BSR
#     if value is not None:
#         if BSR != value:
#             BSR = value
#     return [html.Div(id='loop_breaker_3', children=True)]

# @app.callback(dash.dependencies.Output('BeltSpinRatio', 'value'),
#     [dash.dependencies.Input('loop_breaker_3', 'children')])
# def update_CVResGraph(val):
#     global BSR
#     return BSR

@app.callback([dash.dependencies.Output('FPM', 'value'),
     dash.dependencies.Output('loop_breaker_container_3','children')],
	[dash.dependencies.Input('FibresPerMeter','value')])
def update_CVResGraph(value):
    global SPM, BSR, bsr_min, bsr_max

    if value is not None:
        if baseweight_text == "Switch to Overall Analysis":
            if value < spm_min:
                BSR = emw/(titer * spm_min)   
            elif value >= spm_max:
                BSR = emw/(titer * spm_max)
            else:
                BSR = emw/(titer * value)

    SPM = value
    return [SPM, [html.Div(id='loop_breaker_3', children=True)]] 

@app.callback(dash.dependencies.Output('BeltSpinRatio', 'value'),
    [dash.dependencies.Input('loop_breaker_3', 'children')])
def update_CVResGraph(val):
    global BSR
    return round(BSR,4)

# @app.callback(dash.dependencies.Output('loop_breaker_container_4', 'children'),
# 	[dash.dependencies.Input('FPM','value')])
# def update_loop_breaker(value):
#     global SPM
#     if value is not None:
#         if SPM != value:
#             SPM = value
#     return [html.Div(id='loop_breaker_4', children=True)]

# @app.callback(dash.dependencies.Output('FibresPerMeter', 'value'),
#     [dash.dependencies.Input('loop_breaker_4', 'children')])
# def update_CVResGraph(val):
#     global SPM
#     return SPM

@app.callback(dash.dependencies.Output('emw', 'value'),
	[dash.dependencies.Input('BeltSpinRatio','value'),
    dash.dependencies.Input('FibresPerMeter','value'),
    dash.dependencies.Input('titer', 'value')
    ])
def updateMeanWeigt1(value, value1, value2):
    global emw, SPM, BSR, titer, baseweight_text

    if baseweight_text == "Switch to BaseWeight Analysis":
            if value is not None and value1 is not None and value2 is not None:
                if value2 != titer:
                    titer = value2
                emw = titer * value * value1

    return '%.3f'%(emw)


@app.callback(dash.dependencies.Output('Sigma11', 'value'),
	[dash.dependencies.Input('Sigma_11','value')])
def update_CVResGraph(value):
    global Sigma1
    Sigma1 = value
    return value

@app.callback(dash.dependencies.Output('Sigma21', 'value'),
	[dash.dependencies.Input('Sigma_21','value')])
def update_CVResGraph(value):
    global Sigma2
    Sigma2 = value
    return value

@app.callback(dash.dependencies.Output('A_1', 'value'),
	[dash.dependencies.Input('A1','value')])
def update_CVResGraph(value):
    global A
    A = value
    return value

@app.callback(dash.dependencies.Output('BSR1', 'value'),
	[dash.dependencies.Input('BeltSpinRatio1','value')])
def update_CVResGraph(value):
    global BSR
    BSR = value
    return value

@app.callback(dash.dependencies.Output('FPM1', 'value'),
	[dash.dependencies.Input('FibresPerMeter1','value')])
def update_CVResGraph(value):
    global SPM
    SPM = value
    return value

@app.callback(dash.dependencies.Output('table', 'data'),
    [dash.dependencies.Input('ADD','n_clicks'),
     dash.dependencies.Input('REMOVE','n_clicks'),
     dash.dependencies.Input('CLEAR','n_clicks')])
def updateTable(n_clicks1, n_clicks2, n_clicks3):
    global tableList, Sigma1,Sigma2,A,BSR,SPM,previous_clicks_add_t,previous_clicks_remove_t,previous_clicks_clear_t,trace_count
    if  n_clicks1 > previous_clicks_add_t:
        previous_clicks_add_t = n_clicks1
        tableList.append({'Trace_Number': trace_count, 'Sigma_1': Sigma1, 'Sigma_2': Sigma2, 'A': A, 'BeltSpinRatio': BSR, 'FibresPerMeter': SPM})
        trace_count+=1
        return tableList
    if  n_clicks2 > previous_clicks_remove_t:
        previous_clicks_remove_t = n_clicks2
        tableList.pop()
        if trace_count >=0:
            trace_count-=1
        return tableList
    if  n_clicks3 > previous_clicks_clear_t:
        previous_clicks_clear_t = n_clicks3
        tableList.clear()
        if trace_count >=0:
            trace_count=0
        return tableList
    return tableList

# @app.callback([dash.dependencies.Output('switch', 'children'),
#     dash.dependencies.Output('CVRes_graph', 'figure')],
# 	[dash.dependencies.Input('power','on')])
# def update_power_button(value):
#     if value is not None:
#         if value == True:
#             return ["Multiple Resolutions", []]
#         elif value == False:
#             return ["Single Resolution", []]
#         return ["Multiple Resolutions", []]

@app.callback([dash.dependencies.Output('CVRes_graph', 'figure'),
     dash.dependencies.Output('Cli_graph', 'figure'),
     dash.dependencies.Output('switch', 'children'),
     dash.dependencies.Output('bwa_', 'children'),
     dash.dependencies.Output('BeltSpinRatio', 'min'),
     dash.dependencies.Output('BeltSpinRatio', 'max'),
     dash.dependencies.Output('FibresPerMeter', 'min'),
     dash.dependencies.Output('FibresPerMeter', 'max')],
    [dash.dependencies.Input('Sigma_1','value'),
     dash.dependencies.Input('Sigma_2','value'),
     dash.dependencies.Input('A','value'),
     dash.dependencies.Input('BeltSpinRatio','value'),
     dash.dependencies.Input('FibresPerMeter','value'),
     dash.dependencies.Input('ADD','n_clicks'),
     dash.dependencies.Input('REMOVE','n_clicks'),
     dash.dependencies.Input('CLEAR','n_clicks'),
     dash.dependencies.Input('power','on'),
     dash.dependencies.Input('image','on'),
     dash.dependencies.Input('emw', 'value')])
def updateParamTunerGraph(val1, val2, val3, val4, val5, n_clicks1, n_clicks2, n_clicks3,on, bwa, emw_val):
    global InputList,Sigma1,Sigma2,A,BSR,SPM,previous_clicks_add,previous_clicks_remove, previous_clicks_clear,TuneParams
    global CVRes_graph_curr,cli_graph_curr,df,individual_resolutions_on, baseweight_text, bsr_min, bsr_max, spm_min, spm_max, emw

    # if emw_val:
    #     emw = emw_val

    if  n_clicks1 > previous_clicks_add:
        previous_clicks_add = n_clicks1
        TuneParams = 0
        InputList.append([Sigma1,Sigma2,A,BSR,SPM])
        CVRes_graph_curr = VisAnaObject.getCVtoResolutionGraphForComparision(MLUtils.getPredictionList(InputList))
        cli_graph_curr = VisAnaObject.getCliGraphForComparision(MLUtils.getCliPredictionList(InputList))
    if  n_clicks2 > previous_clicks_remove:
        previous_clicks_remove = n_clicks2
        InputList.pop()
        CVRes_graph_curr = VisAnaObject.getCVtoResolutionGraphForComparision(MLUtils.getPredictionList(InputList))
        cli_graph_curr = VisAnaObject.getCliGraphForComparision(MLUtils.getCliPredictionList(InputList))
    if  n_clicks3 > previous_clicks_clear:
        previous_clicks_clear = n_clicks3
        TuneParams = 1
        InputList.clear()
        CVRes_graph_curr = VisAnaObject.getCVtoResolutionGraphForComparision(MLUtils.getPredictionList(InputList))
        cli_graph_curr = VisAnaObject.getCliGraphForComparision(MLUtils.getCliPredictionList(InputList))
    if TuneParams > 0:
        CVRes_graph_curr = VisAnaObject.getCVtoResolutionGraph(MLUtils.getPredictions([Sigma1,Sigma2,A,BSR,SPM]))
        cliPrediction = MLUtils.CliPrediction([Sigma1,Sigma2,A,BSR,SPM])
        # print("Cli prediction", cliPrediction)
        #CVRes_graph_curr.add_trace(go.Bar(x=[60], y=[cliPrediction[0]*100], name="Cli"))
        cli_graph_curr = VisAnaObject.getCliGraph(cliPrediction[0])

    if bwa is not None:
        if bwa == True:
            baseweight_text = "Switch to BaseWeight Analysis"
            bsr_min = 0.01
            bsr_max = 0.25
            spm_min = 200
            spm_max = 10000

        elif bwa == False:
            baseweight_text = "Switch to Overall Analysis"
            bsr_min = max(0.01, ((emw/titer)/10000))
            bsr_max = min(0.25, ((emw/titer)/200))
            spm_min = max(200, ((emw/titer)/0.25))
            spm_max = min(10000, ((emw/titer)/0.01))

    if on is not None:
        if on == True:
            if(len(InputList) != 0):
                CVRes_graph_curr = VisAnaObject.getCVtoResolutionGraphForComparision(MLUtils.getPredictionList(InputList))
                cli_graph_curr = VisAnaObject.getCliGraphForComparision(MLUtils.getCliPredictionList(InputList))
            else:
                CVRes_graph_curr = VisAnaObject.getCVtoResolutionGraph(MLUtils.getPredictions([Sigma1,Sigma2,A,BSR,SPM]))
                cliPrediction = MLUtils.CliPrediction([Sigma1,Sigma2,A,BSR,SPM])
                #CVRes_graph_curr.add_trace(go.Bar(x=[60], y=[cliPrediction[0]*100], name="Cli"))
                cli_graph_curr = VisAnaObject.getCliGraph(cliPrediction[0])
            individual_resolutions_on = "Multiple grid-sizes"
        elif on == False:
            currDataFrame = list()
            currPrediction = list()
            currCliPrediction = list()

            currDataFrameSigma_1 = DataPrepUtils.getDataFrame("Sigma_1", Sigma2, A, BSR, SPM, 100)
            currPredictionSigma_1 = DataPrepUtils.getPredictions(currDataFrameSigma_1)
            currDataFrame.append(currDataFrameSigma_1)
            currPrediction.append(currPredictionSigma_1)
            currCliPrediction.append(DataPrepUtils.getCliPredictions(currDataFrameSigma_1))

            currDataFrameSigma_2 = DataPrepUtils.getDataFrame("Sigma_2", Sigma1, A, BSR, SPM, 100)
            currPredictionSigma_2 = DataPrepUtils.getPredictions(currDataFrameSigma_2)
            currDataFrame.append(currDataFrameSigma_2)
            currPrediction.append(currPredictionSigma_2)
            currCliPrediction.append(DataPrepUtils.getCliPredictions(currDataFrameSigma_2))

            currDataFrameA = DataPrepUtils.getDataFrame("A", Sigma1, Sigma2, BSR, SPM, 100)
            currPredictionA = DataPrepUtils.getPredictions(currDataFrameA)
            currDataFrame.append(currDataFrameA)
            currPrediction.append(currPredictionA)
            currCliPrediction.append(DataPrepUtils.getCliPredictions(currDataFrameA))

            currDataFrameBSR = DataPrepUtils.getDataFrame("BeltSpinRatio", Sigma1, Sigma2, A, SPM, 100)
            currPredictionBSR = DataPrepUtils.getPredictions(currDataFrameBSR)
            currDataFrame.append(currDataFrameBSR)
            currPrediction.append(currPredictionBSR)
            currCliPrediction.append(DataPrepUtils.getCliPredictions(currDataFrameBSR))

            currDataFrameSPM = DataPrepUtils.getDataFrame("SpinPositionsPerMeterInverse", Sigma1, Sigma2, A, BSR, 100)
            currPredictionSPM = DataPrepUtils.getPredictions(currDataFrameSPM)
            currDataFrame.append(currDataFrameSPM)
            currPrediction.append(currPredictionSPM)
            currCliPrediction.append(DataPrepUtils.getCliPredictions(currDataFrameSPM))

            CVRes_graph_curr = VisAnaObject.getCVForIndividualResolutions(currPrediction, currCliPrediction, currDataFrame)
            cli_graph_curr = VisAnaObject.getEmptyPlot()
            individual_resolutions_on = "Individual grid-sizes"
    return [CVRes_graph_curr, cli_graph_curr, individual_resolutions_on, baseweight_text, bsr_min, bsr_max, spm_min, spm_max]

@app.callback(dash.dependencies.Output('Surface_graph', 'figure'),
	[dash.dependencies.Input('Sigma_11','value'),
     dash.dependencies.Input('Sigma_21','value'),
     dash.dependencies.Input('A1','value'),
     dash.dependencies.Input('BeltSpinRatio1','value'),
     dash.dependencies.Input('FibresPerMeter1','value'),
     dash.dependencies.Input('HEAT','n_clicks'),
     dash.dependencies.Input('resolution','value'),
     dash.dependencies.Input('min_1', 'value'),
     dash.dependencies.Input('min_2', 'value'),
     dash.dependencies.Input('max_1', 'value'),
     dash.dependencies.Input('max_2', 'value')])
def update_3DSGraph(val1, val2, val3, val4, val5, n_clicks, res, min1, min2, max1, max2):
    global previous_clicks_heatmap,Sigma1,Sigma2,A,BSR,SPM,fullRangePredictionsGlobal, Surface_graph_global,Resolution
    global subrange_min_1,subrange_max_1,subrange_min_2,subrange_max_2

    if min1 is not None: 
        subrange_min_1 = min1

    if max1 is not None: 
        subrange_max_1 = max1

    if min2 is not None: 
        subrange_min_2 = min2

    if max2 is not None: 
        subrange_max_2 = max2

    if res is not None:
        fullRangePredictionsGlobal = DataPrepUtils.getFullRangePredictions('S1S2', A, BSR, SPM,25,res,subrange_min_1,subrange_max_1,subrange_min_2,subrange_max_2)
        Surface_graph_global = VisAnaObject.get3DSurfacePlot('Sigma_1 and Sigma_2', fullRangePredictionsGlobal, res)

    if n_clicks > previous_clicks_heatmap:
        previous_clicks_heatmap = n_clicks
        Surface_graph_global = VisAnaObject.getHeatMap(frozen_input, fullRangePredictionsGlobal)
        # return Surface_graph_curr
    if frozen_input == 'S1S2':
        fullRangePredictionsGlobal = DataPrepUtils.getFullRangePredictions('S1S2', A, BSR, SPM,25,Resolution,subrange_min_1,subrange_max_1,subrange_min_2,subrange_max_2)
        Surface_graph_global = VisAnaObject.get3DSurfacePlot('Sigma_1 and Sigma_2', fullRangePredictionsGlobal, res)
        # return Surface_graph_curr
    if frozen_input == 'S1A':
        fullRangePredictionsGlobal = DataPrepUtils.getFullRangePredictions('S1A', Sigma2,BSR, SPM,25,Resolution,subrange_min_1,subrange_max_1,subrange_min_2,subrange_max_2)
        Surface_graph_global = VisAnaObject.get3DSurfacePlot('Sigma_1 and A', fullRangePredictionsGlobal, res)
        # return Surface_graph_curr
    if frozen_input == 'S1BSR':
        fullRangePredictionsGlobal = DataPrepUtils.getFullRangePredictions('S1BSR', Sigma2, A, SPM,25,Resolution,subrange_min_1,subrange_max_1,subrange_min_2,subrange_max_2)
        Surface_graph_global = VisAnaObject.get3DSurfacePlot('Sigma_1 and BeltSpinRatio', fullRangePredictionsGlobal, res)
        # return Surface_graph_curr
    if frozen_input == 'S1SPM':
        fullRangePredictionsGlobal = DataPrepUtils.getFullRangePredictions('S1SPM', Sigma2, A, BSR,25,Resolution,subrange_min_1,subrange_max_1,subrange_min_2,subrange_max_2)
        Surface_graph_global = VisAnaObject.get3DSurfacePlot('Sigma_1 and SpinPositionsPerMeter', fullRangePredictionsGlobal, res)
        # return Surface_graph_curr
    if frozen_input == 'S2A':
        fullRangePredictionsGlobal = DataPrepUtils.getFullRangePredictions('S2A', Sigma1, BSR, SPM, 25,Resolution,subrange_min_1,subrange_max_1,subrange_min_2,subrange_max_2)
        Surface_graph_global = VisAnaObject.get3DSurfacePlot('Sigma_2 and A', fullRangePredictionsGlobal, res)
        # return Surface_graph_curr
    if frozen_input == 'S2BSR':
        fullRangePredictionsGlobal = DataPrepUtils.getFullRangePredictions('S2BSR', Sigma1, A, SPM,25,Resolution,subrange_min_1,subrange_max_1,subrange_min_2,subrange_max_2)
        Surface_graph_global = VisAnaObject.get3DSurfacePlot('Sigma_2 and BeltSpinRatio', fullRangePredictionsGlobal, res)
        # return Surface_graph_curr
    if frozen_input == 'S2SPM':
        fullRangePredictionsGlobal = DataPrepUtils.getFullRangePredictions('S2SPM', Sigma1, A, BSR,25,Resolution,subrange_min_1,subrange_max_1,subrange_min_2,subrange_max_2)
        Surface_graph_global = VisAnaObject.get3DSurfacePlot('Sigma_2 and SpinPositionsPerMeter', fullRangePredictionsGlobal, res)
        # return Surface_graph_curr
    if frozen_input == 'ABSR':
        fullRangePredictionsGlobal = DataPrepUtils.getFullRangePredictions('ABSR', Sigma1, Sigma2, SPM,25,Resolution,subrange_min_1,subrange_max_1,subrange_min_2,subrange_max_2)
        Surface_graph_global = VisAnaObject.get3DSurfacePlot('A and BeltSpinRatio', fullRangePredictionsGlobal, res)
        # return Surface_graph_curr
    if frozen_input == 'ASPM':
        fullRangePredictionsGlobal = DataPrepUtils.getFullRangePredictions('ASPM', Sigma1, Sigma2, BSR,25,Resolution,subrange_min_1,subrange_max_1,subrange_min_2,subrange_max_2)
        Surface_graph_global = VisAnaObject.get3DSurfacePlot('A and SpinPositionsPerMeter', fullRangePredictionsGlobal, res)
        # return Surface_graph_curr
    if frozen_input == 'BSRSPM':
        fullRangePredictionsGlobal = DataPrepUtils.getFullRangePredictions('BSRSPM', Sigma1, Sigma2, A,25,Resolution,subrange_min_1,subrange_max_1,subrange_min_2,subrange_max_2)
        Surface_graph_global = VisAnaObject.get3DSurfacePlot('BeltSpinRatio and SpinPositionsPerMeter', fullRangePredictionsGlobal, res)
        # return Surface_graph_curr
    return Surface_graph_global
         


@app.callback([dash.dependencies.Output('Sigma_11', 'style'),
    dash.dependencies.Output('Sigma_21', 'style'),
    dash.dependencies.Output('A1', 'style'),
    dash.dependencies.Output('BeltSpinRatio1', 'style'),
    dash.dependencies.Output('FibresPerMeter1', 'style'),
    dash.dependencies.Output('option_1', 'children'),
    dash.dependencies.Output('option_2', 'children'),
    dash.dependencies.Output('min_1', 'value'),
    dash.dependencies.Output('max_1', 'value'),
    dash.dependencies.Output('min_2', 'value'),
    dash.dependencies.Output('max_2', 'value')],
	[dash.dependencies.Input('Inputs','value'),
    dash.dependencies.Input('resolution','value')])
def highlightKnob(value, res):
    global frozen_input, Resolution, subrange_min_1,subrange_min_2,subrange_max_1,subrange_max_2
    if res is not None:
        Resolution = res
    if value =='S1S2':
        frozen_input = 'S1S2'
        subrange_min_1 = 1
        subrange_max_1 = 50
        subrange_min_2 = 1
        subrange_max_2 = 50
        return {"box-shadow": "5px 10px 1px black"},{"box-shadow": "5px 10px 1px black"},{},{},{}, "Sigma_1", "Sigma_2", 1,50,1,50
    if value == 'S1A':
        frozen_input = 'S1A'
        subrange_min_1 = 1
        subrange_max_1 = 50
        subrange_min_2 = 1
        subrange_max_2 = 50
        return {"box-shadow": "5px 10px 1px black"},{},{"box-shadow": "5px 10px 1px black"},{},{}, "Sigma_1", "A", 1,50,1,50
    if value == 'S1BSR':
        frozen_input = 'S1BSR'
        subrange_min_1 = 1
        subrange_max_1 = 50
        subrange_min_2 = 0.01
        subrange_max_2 = 0.25
        return {"box-shadow": "5px 10px 1px black"},{},{},{"box-shadow": "5px 10px 1px black"},{}, "Sigma_1", "BeltSpinRatio", 1,50,0.01,0.25
    if value == 'S1SPM':
        frozen_input = 'S1SPM'
        subrange_min_1 = 1
        subrange_max_1 = 50
        subrange_min_2 = 200
        subrange_max_2 = 10000
        return {"box-shadow": "5px 10px 1px black"},{},{},{},{"box-shadow": "5px 10px 1px black"}, "Sigma_1", "SpinPosPerMeter", 1,50,200,10000
    if value == 'S2A':
        frozen_input = 'S2A'
        subrange_min_1 = 1
        subrange_max_1 = 50
        subrange_min_2 = 1
        subrange_max_2 = 50
        return {},{"box-shadow": "5px 10px 1px black"},{"box-shadow": "5px 10px 1px black"},{},{}, "Sigma_2", "A", 1,50,1,50
    if value =='S2BSR':
        frozen_input = 'S2BSR'
        subrange_min_1 = 1
        subrange_max_1 = 50
        subrange_min_2 = 0.01
        subrange_max_2 = 0.25
        return {},{"box-shadow": "5px 10px 1px black"},{},{"box-shadow": "5px 10px 1px black"},{}, "Sigma_2", "BeltSpinRatio", 1,50,0.01,0.25
    if value == 'S2SPM':
        frozen_input = 'S2SPM'
        subrange_min_1 = 1
        subrange_max_1 = 50
        subrange_min_2 = 200
        subrange_max_2 = 10000
        return {},{"box-shadow": "5px 10px 1px black"},{},{},{"box-shadow": "5px 10px 1px black"}, "Sigma_2", "SpinPosPerMeter", 1,50,200,10000
    if value == 'ABSR':
        frozen_input = 'ABSR'
        subrange_min_1 = 1
        subrange_max_1 = 50
        subrange_min_2 = 0.01
        subrange_max_2 = 0.25
        return {},{},{"box-shadow": "5px 10px 1px black"},{"box-shadow": "5px 10px 1px black"},{}, "A", "BeltSpinRatio", 1,50,0.01,0.25
    if value == 'ASPM':
        frozen_input = 'ASPM'
        subrange_min_1 = 1
        subrange_max_1 = 50
        subrange_min_2 = 200
        subrange_max_2 = 10000
        return {},{},{"box-shadow": "5px 10px 1px black"},{},{"box-shadow": "5px 10px 1px black"}, "A", "SpinPosPerMeter", 1,50,200,10000
    if value == 'BSRSPM':
        frozen_input = 'BSRSPM'
        subrange_min_1 = 0.01
        subrange_max_1 = 0.25
        subrange_min_2 = 200
        subrange_max_2 = 10000
        return {},{},{},{"box-shadow": "5px 10px 1px black"},{"box-shadow": "5px 10px 1px black"}, "BeltSpinRatio", "SpinPosPerMeter",0.01,0.25,200,10000
    frozen_input = 'None'
    return {},{},{},{},{},"Sigma_1", "Sigma_2", 1,50,1,50

@app.callback([dash.dependencies.Output('InputSpaceGraph', 'figure'),
    dash.dependencies.Output('OutputSpaceGraph', 'figure'),
    dash.dependencies.Output('ParallelCoordinatesGraph', 'figure')],
	[dash.dependencies.Input('FP_Samples','n_clicks'),
    dash.dependencies.Input('InputSpaceGraph','clickData'),
    dash.dependencies.Input('FP', 'n_clicks')])
def updateInputSpaceGraph(n_clicks,clickData, n_clicks1):
    global prevSampleGenerationClicks, dframe3, FocalPoint,FocalPointData, pc,pc1, scaler, scaler1, ParallelCoordinatesGraph, tick_box_array, InputSpaceGraph, OutputSpaceGraph
    global curr_number_input_1, curr_number_input_2, curr_number_input_3, curr_number_input_4, curr_number_input_5, prevUserFocalPoint, prevIndex
    if n_clicks > prevSampleGenerationClicks:
        prevSampleGenerationClicks = n_clicks
        FocalPointCurr = dframe3.iloc[FocalPoint]
        sigma1_curr = FocalPointCurr.Sigma_1
        sigma2_curr = FocalPointCurr.Sigma_2
        a_curr = FocalPointCurr.A
        bsr_curr = FocalPointCurr.BeltSpinRatio
        spm_curr = FocalPointCurr.SpinPositionsPerMeterInverse

        current_array = np.zeros(5)

        for i in tick_box_array:
            current_array[i-1] = i

        feat_sigma_1 = DataPrepUtils.getCheckBoxValues(current_array[0], sigma1_curr,  max(1, float(sigma1_curr) - 4.9), min(50, float(sigma1_curr) + 4.9), 50)
        feat_sigma_2 = DataPrepUtils.getCheckBoxValues(current_array[1], sigma2_curr,  max(1, float(sigma2_curr) - 4.9), min(50, float(sigma2_curr) + 4.9), 50)
        feat_a = DataPrepUtils.getCheckBoxValues(current_array[2], a_curr,  max(1, float(a_curr) - 4.9), min(50, float(a_curr) + 4.9), 50)
        feat_bsr = DataPrepUtils.getCheckBoxValues(current_array[3], bsr_curr,  max(0.01, float(bsr_curr) - 0.024), min(0.25, float(bsr_curr) + 0.024), 50)
        feat_spm = DataPrepUtils.getCheckBoxValues(current_array[4], spm_curr,  max(200, float(spm_curr) - 980), min(10000, float(spm_curr) + 980), 50)


        dataframe = pd.DataFrame(columns = ["Sigma_1", "Sigma_2", "A", "BeltSpinRatio", "SpinPositionsPerMeterInverse"])
        dataframe['Sigma_1'] = feat_sigma_1
        dataframe['Sigma_2'] = feat_sigma_2
        dataframe['A'] = feat_a
        dataframe['BeltSpinRatio'] = feat_bsr
        dataframe['SpinPositionsPerMeterInverse'] = feat_spm

        z_curr = pc.transform(scaler.transform(dataframe))

        pred = DataPrepUtils.getOutputSpacePrediction(dataframe)

        z1_curr = pc1.transform(scaler1.transform(pred)) 

        samples_data_frame = pd.DataFrame()
        samples_data_frame['Sigma_1'] = feat_sigma_1
        samples_data_frame['Sigma_2'] = feat_sigma_2
        samples_data_frame['A'] = feat_a
        samples_data_frame['BeltSpinRatio'] = feat_bsr
        samples_data_frame['SpinPositionsPerMeterInverse'] = feat_spm
        samples_data_frame['0.5'] = pred[:,0]
        samples_data_frame['1'] = pred[:,1]
        samples_data_frame['2'] = pred[:,2]
        samples_data_frame['5'] = pred[:,3]
        samples_data_frame['10'] = pred[:,4]
        samples_data_frame['20'] = pred[:,5]
        samples_data_frame['50'] = pred[:,6]
        samples_data_frame['pc1_ip'] = z_curr[:, 0]
        samples_data_frame['pc2_ip'] = z_curr[:, 1]
        samples_data_frame['pc1_op'] = z1_curr[:, 0]
        samples_data_frame['pc2_op'] = z1_curr[:, 1] 
        samples_data_frame['Label_op'] = gmm.predict(z1_curr)

        dframe3 = dframe3.append(samples_data_frame)

        InputSpaceGraph = px.scatter(dframe3, x='pc1_ip', y='pc2_ip', color = 'Label_op',
                            hover_data = {'Sigma_1' : True,
                                          'Sigma_2' : True,
                                          'A' : True,
                                          'BeltSpinRatio' : True,
                                          'SpinPositionsPerMeterInverse' : True,
                                          'pc1_ip' : False,
                                          'pc2_ip' : False,
                                          'Label_op': False
                                         })

        OutputSpaceGraph = px.scatter(dframe3, x='pc1_op', y='pc2_op',color = 'Label_op',
                            hover_data = {'0.5' : True,
                                          '1' : True,
                                          '2' : True,
                                          '5' : True,
                                          '10' : True,
                                          '20' : True,
                                          '50' : True,
                                          'pc1_op' : False,
                                          'pc2_op' : False,
                                          'Label_op': False
                                         })
        
        array = np.array([sigma1_curr, sigma2_curr, a_curr, bsr_curr, spm_curr])
        array = array.reshape(1, -1)
        pred_temp = DataPrepUtils.getOutputSpacePrediction(array)
        df_temp_ip = list()
        df_temp_ip.append({'Sigma_1':sigma1_curr, 'Sigma_2': sigma2_curr, 'A': a_curr, 'BSR': bsr_curr, 'SPM': spm_curr})
        df_temp_ip = pd.DataFrame(df_temp_ip)

        df_temp_op = list()
        df_temp_op.append({'Res0.5': pred_temp[:,0], 'Res1': pred_temp[:,1], 'Res2': pred_temp[:,2], 'Res5': pred_temp[:,3], 
                        'Res10':pred_temp[:,4], 'Res20':pred_temp[:,5], 'Res50':pred_temp[:,6]})
        df_temp_op = pd.DataFrame(df_temp_op)

        InputSpaceGraph.add_trace(go.Scatter(x=[InputSpaceGraph['data'][0]['x'][FocalPoint]], y=[InputSpaceGraph['data'][0]['y'][FocalPoint]],
                    mode='markers', marker=dict(color='red', size=10),
                    customdata=df_temp_ip,
                    hovertemplate='<b>Sigma_1:%{customdata[0]:.3f}</b><br>Sigma_2:%{customdata[1]:.3f} <br>A: %{customdata[2]:.3f}' +
                    '<br>BSR: %{customdata[3]:.3f} <br>SPM: %{customdata[3]:.3f}'))

        OutputSpaceGraph.add_trace(go.Scatter(x=[OutputSpaceGraph['data'][0]['x'][FocalPoint]], y=[OutputSpaceGraph['data'][0]['y'][FocalPoint]],
                    mode='markers', marker=dict(color='red', size=10),
                    customdata=df_temp_op,
                    hovertemplate='<b>Res0.5:%{customdata[0]:.3f}</b><br>Res1:%{customdata[1]:.3f} <br>Res2: %{customdata[2]:.3f}' +
                    '<br>Res5: %{customdata[3]:.3f} <br>Res10: %{customdata[4]:.3f} <br>Res20: %{customdata[5]:.3f} <br>Res50: %{customdata[6]:.3f} '))

        dataframeFull = dataframe.copy()

        dataframeFull['0.5'] = pred[:,0]
        dataframeFull['1'] = pred[:,1]
        dataframeFull['2'] = pred[:,2]
        dataframeFull['5'] = pred[:,3]
        dataframeFull['10'] = pred[:,4]
        dataframeFull['20'] = pred[:,5]
        dataframeFull['50'] = pred[:,6]

        CurrAttrib = ParallelCoordinatesGraph['data'][0]

        if n_clicks > 1:
            for i in range(0,12):
                ParallelCoordinatesGraph['data'][0].dimensions[i].values = CurrAttrib.dimensions[i].values[:-50]

        columns = np.array(dataframeFull.columns)

        for i in range(0,12):
            ParallelCoordinatesGraph['data'][0].dimensions[i].values = np.concatenate((CurrAttrib.dimensions[i].values,dataframeFull[columns[i]]))


    if clickData is not None:
        index = clickData['points'][0]['pointIndex']
        if(prevIndex != index):
            prevIndex = index
            InputSpaceGraph.data = [InputSpaceGraph.data[0]]
            OutputSpaceGraph.data = [OutputSpaceGraph.data[0]]

            FocalPointData = clickData['points'][0]
            FocalPoint = index
            ParallelCoordinatesGraph['data'][0].dimensions[0].constraintrange = [dframe3.iloc[index].Sigma_1, dframe3.iloc[index].Sigma_1+0.001]
            ParallelCoordinatesGraph['data'][0].dimensions[1].constraintrange = [dframe3.iloc[index].Sigma_2, dframe3.iloc[index].Sigma_2+0.001]
            ParallelCoordinatesGraph['data'][0].dimensions[2].constraintrange = [dframe3.iloc[index].A, dframe3.iloc[index].A+0.001]
            ParallelCoordinatesGraph['data'][0].dimensions[3].constraintrange = [dframe3.iloc[index].BeltSpinRatio, dframe3.iloc[index].BeltSpinRatio+0.001]
            ParallelCoordinatesGraph['data'][0].dimensions[4].constraintrange = [dframe3.iloc[index].SpinPositionsPerMeterInverse, dframe3.iloc[index].SpinPositionsPerMeterInverse+0.001]

            df_temp = list()
            df_temp.append({'Sigma_1': dframe3.iloc[FocalPoint].Sigma_1, 'Sigma_2': dframe3.iloc[FocalPoint].Sigma_2, 
                        'A': dframe3.iloc[FocalPoint].A, 'BeltSpinRatio': dframe3.iloc[FocalPoint].BeltSpinRatio, 
                        'SpinPositionsPerMeterInverse':dframe3.iloc[FocalPoint].SpinPositionsPerMeterInverse})
            df_temp = pd.DataFrame(df_temp)
            InputSpaceGraph.add_trace(go.Scatter(x=[clickData['points'][0]['x']], y=[clickData['points'][0]['y']],
                    mode='markers', marker=dict(color='red', size=10),
                    customdata=df_temp,
                    hovertemplate='<b>Sigma_1:%{customdata[0]:.3f}</b><br>Sigma_2:%{customdata[1]:.3f} <br>A: %{customdata[2]:.3f}' +
                    '<br>BeltSpinRatio: %{customdata[3]:.3f} <br>SpinPositionsPerMeterInverse: %{customdata[4]:.3f} '))


            array = np.array([dframe3.iloc[FocalPoint].Sigma_1, dframe3.iloc[FocalPoint].Sigma_2, dframe3.iloc[FocalPoint].A, 
                        dframe3.iloc[FocalPoint].BeltSpinRatio, dframe3.iloc[FocalPoint].SpinPositionsPerMeterInverse])
            array = array.reshape(1, -1)
            pred_temp = DataPrepUtils.getOutputSpacePrediction(array)

            df_temp_op = list()
            df_temp_op.append({'Res0.5': pred_temp[:,0], 'Res1': pred_temp[:,1], 'Res2': pred_temp[:,2], 'Res5': pred_temp[:,3], 
                        'Res10':pred_temp[:,4], 'Res20':pred_temp[:,5], 'Res50':pred_temp[:,6]})
            df_temp_op = pd.DataFrame(df_temp_op)

            OutputSpaceGraph.add_trace(go.Scatter(x=[OutputSpaceGraph['data'][0]['x'][FocalPoint]], y=[OutputSpaceGraph['data'][0]['y'][FocalPoint]],
                    mode='markers', marker=dict(color='red', size=10),
                    customdata=df_temp_op,
                    hovertemplate='<b>Res0.5:%{customdata[0]:.3f}</b><br>Res1:%{customdata[1]:.3f} <br>Res2: %{customdata[2]:.3f}' +
                    '<br>Res5: %{customdata[3]:.3f} <br>Res10: %{customdata[4]:.3f} <br>Res20: %{customdata[5]:.3f} <br>Res50: %{customdata[6]:.3f} '))


    if n_clicks1 > prevUserFocalPoint:
        prevUserFocalPoint = n_clicks1
        
        df_temp_1 = list() 
        df_temp_1.append({'Sigma_1': curr_number_input_1, 'Sigma_2': curr_number_input_2, 
                        'A': curr_number_input_3, 'BeltSpinRatio': curr_number_input_4, 
                        'SpinPositionsPerMeterInverse':curr_number_input_5})
        df_temp_1 = pd.DataFrame(df_temp_1)
        df_temp_copy = df_temp_1.copy()

        z_curr = pc.transform(scaler.transform(df_temp_1))

        df_temp_1['pc_1'] = z_curr[:, 0]
        df_temp_1['pc_2'] = z_curr[:, 1]


        pred_temp = DataPrepUtils.getOutputSpacePrediction(df_temp_copy)
        df_temp_op = list()
        df_temp_op.append({'Res: 0.5': pred_temp[:,0][0], 'Res: 1': pred_temp[:,1][0], 'Res: 2': pred_temp[:,2][0], 'Res: 5': pred_temp[:,3][0], 
                        'Res: 10':pred_temp[:,4][0], 'Res: 20':pred_temp[:,5][0], 'Res: 50':pred_temp[:,6][0]})
        df_temp_op = pd.DataFrame(df_temp_op)

        z1_curr = pc1.transform(scaler1.transform(df_temp_op))

        df_temp_op['pc_1'] = z1_curr[:, 0]
        df_temp_op['pc_2'] = z1_curr[:, 1]

        newRow = list()
        newRow.append({'Sigma_1': curr_number_input_1, 'Sigma_2': curr_number_input_2, 
                        'A': curr_number_input_3, 'BeltSpinRatio': curr_number_input_4, 
                        'SpinPositionsPerMeterInverse':curr_number_input_5,
                        '0.5': pred_temp[:,0][0], '1': pred_temp[:,1][0], '2': pred_temp[:,2][0], '5': pred_temp[:,3][0], 
                        '10':pred_temp[:,4][0], '20':pred_temp[:,5][0], '50':pred_temp[:,6][0],
                        'pc1_ip': z_curr[:, 0][0], 'pc2_ip': z_curr[:, 1][0], 'pc1_op': z1_curr[:, 0][0], 'pc2_op': z1_curr[:, 1][0],
                        'Label_op': gmm.predict(z1_curr)[0]})
        
        newRow = pd.DataFrame(newRow)
        dframe3 = dframe3.append(newRow)
        FocalPoint = len(dframe3) - 1
        


        InputSpaceGraph = px.scatter(dframe3, x='pc1_ip', y='pc2_ip', color = 'Label_op',
                            hover_data = {'Sigma_1' : True,
                                          'Sigma_2' : True,
                                          'A' : True,
                                          'BeltSpinRatio' : True,
                                          'SpinPositionsPerMeterInverse' : True,
                                          'pc1_ip' : False,
                                          'pc2_ip' : False,
                                          'Label_op': False,
                                         })

        InputSpaceGraph.add_trace(go.Scatter(x=z_curr[:, 0], y=z_curr[:, 1],
                    mode='markers', marker=dict(color='black', size=10),
                    customdata=df_temp_1,
                    hovertemplate='<b>Sigma_1:%{customdata[0]:.3f}</b><br>Sigma_2:%{customdata[1]:.3f} <br>A: %{customdata[2]:.3f}' +
                    '<br>BeltSpinRatio: %{customdata[3]:.3f} <br>SpinPositionsPerMeterInverse: %{customdata[4]:.3f} ' +
                    '<br>pc_1: %{customdata[5]:.3f} <br>pc_2: %{customdata[6]:.3f}'))


        OutputSpaceGraph = px.scatter(dframe3, x='pc1_op', y='pc2_op', color = 'Label_op',
                            hover_data = {'0.5' : True,
                                          '1' : True,
                                          '2' : True,
                                          '5' : True,
                                          '10' : True,
                                          '20' : True,
                                          '50' : True,
                                          'pc1_op' : False,
                                          'pc2_op' : False,
                                          'Label_op': False
                                         })

        OutputSpaceGraph.add_trace(go.Scatter(x=z1_curr[:, 0], y=z1_curr[:, 1],
                    mode='markers', marker=dict(color='black', size=10),
                    customdata=df_temp_op,
                    hovertemplate='<b>Res 0.5:%{customdata[0]:.3f}</b><br>Res 1:%{customdata[1]:.3f} <br>Res 2: %{customdata[2]:.3f}' +
                    '<br>Res 5: %{customdata[3]:.3f} <br>Res 10: %{customdata[4]:.3f} <br>Res 20: %{customdata[5]:.3f} <br>Res 50: %{customdata[6]:.3f} ' +
                    '<br>pc_1: %{customdata[7]:.3f} <br>pc_2: %{customdata[8]:.3f}'))
        

        dataframeFull = df_temp_1.copy()

        dataframeFull['0.5'] = pred_temp[:,0]
        dataframeFull['1'] = pred_temp[:,1]
        dataframeFull['2'] = pred_temp[:,2]
        dataframeFull['5'] = pred_temp[:,3]
        dataframeFull['10'] = pred_temp[:,4]
        dataframeFull['20'] = pred_temp[:,5]
        dataframeFull['50'] = pred_temp[:,6]
        del dataframeFull['pc_1']
        del dataframeFull['pc_2']
        columns = np.array(dataframeFull.columns)

        CurrAttrib = ParallelCoordinatesGraph['data'][0]

        for i in range(0,12):
            ParallelCoordinatesGraph['data'][0].dimensions[i].values = np.concatenate((CurrAttrib.dimensions[i].values, dataframeFull[columns[i]]))

        ParallelCoordinatesGraph['data'][0].dimensions[0].constraintrange = [float(curr_number_input_1), float(curr_number_input_1)+0.001]
        ParallelCoordinatesGraph['data'][0].dimensions[1].constraintrange = [float(curr_number_input_2), float(curr_number_input_2)+0.001]
        ParallelCoordinatesGraph['data'][0].dimensions[2].constraintrange = [float(curr_number_input_3), float(curr_number_input_3)+0.001]
        ParallelCoordinatesGraph['data'][0].dimensions[3].constraintrange = [float(curr_number_input_4), float(curr_number_input_4)+0.001]
        ParallelCoordinatesGraph['data'][0].dimensions[4].constraintrange = [float(curr_number_input_5), float(curr_number_input_5)+0.001]

    return InputSpaceGraph, OutputSpaceGraph, ParallelCoordinatesGraph

@app.callback([dash.dependencies.Output('number_input_1', 'value'),
    dash.dependencies.Output('number_input_2', 'value'),
    dash.dependencies.Output('number_input_3', 'value'),
    dash.dependencies.Output('number_input_4', 'value'),
    dash.dependencies.Output('number_input_5', 'value')],
    [dash.dependencies.Input('InputSpaceGraph','clickData')])
def displayClickedData(clickData):
    if clickData is not None:
        index = clickData['points'][0]['pointIndex']
        return ['%.2f'%(dframe3.iloc[index].Sigma_1),'%.2f'%(dframe3.iloc[index].Sigma_2),'%.2f'%(dframe3.iloc[index].A),'%.2f'%(dframe3.iloc[index].BeltSpinRatio),'%.2f'%(dframe3.iloc[index].SpinPositionsPerMeterInverse)]

    # ClickDataCurr = dframe3.iloc[FocalPoint]
    # return [ClickDataCurr.Sigma_1, ClickDataCurr.Sigma_2, ClickDataCurr.A, ClickDataCurr.BeltSpinRatio, ClickDataCurr.SpinPositionsPerMeterInverse]
    return ["Sigma_1","Sigma_2","A","BeltSpinRatio","SpinPositionsPerMeterInverse"]


@app.callback([dash.dependencies.Output('dummy', 'children')],
	[dash.dependencies.Input('tick-box','value'),
     dash.dependencies.Input('number_input_1', 'value'),
     dash.dependencies.Input('number_input_2', 'value'),
     dash.dependencies.Input('number_input_3', 'value'),
     dash.dependencies.Input('number_input_4', 'value'),
     dash.dependencies.Input('number_input_5', 'value')],
    [dash.dependencies.State('dummy', 'children')])
def updateUserDefinedValue(value, value1, value2, value3, value4, value5, state):
    global tick_box_array, curr_number_input_1, curr_number_input_2, curr_number_input_3, curr_number_input_4, curr_number_input_5

    if value is not None:
        tick_box_array = np.sort(np.array(value))

    if value1 != curr_number_input_1:
        curr_number_input_1 = value1

    if value2 != curr_number_input_2:
        curr_number_input_2 = value2

    if value3 != curr_number_input_3:
        curr_number_input_3 = value3

    if value4 != curr_number_input_4:
        curr_number_input_4 = value4

    if value5 != curr_number_input_5:
        curr_number_input_5 = value5

    return [state]
    
@app.callback([dash.dependencies.Output('OutputCluster', 'figure'),
    dash.dependencies.Output('MeanPlot', 'figure'),
    dash.dependencies.Output('boxPlot', 'figure'),
    dash.dependencies.Output('boxPlot1', 'figure'),
    dash.dependencies.Output('boxPlot2', 'figure'),
    dash.dependencies.Output('boxPlot3', 'figure'),
    dash.dependencies.Output('boxPlot4', 'figure')],
	[dash.dependencies.Input('Cluster1','n_clicks'),
    dash.dependencies.Input('Cluster2','n_clicks'),
    dash.dependencies.Input('Cluster3','n_clicks'),
    dash.dependencies.Input('Cluster4','n_clicks'),
    dash.dependencies.Input('Cluster5','n_clicks')])
def updateCluster(n_clicks1, n_clicks2, n_clicks3, n_clicks4, n_clicks5):
    global cluster_click1, cluster_click2, cluster_click3, cluster_click4, cluster_click5,meanPlot,boxPlot,boxPlot1, boxPlot2, boxPlot3, boxPlot4
    clusterPlot = VisAnaObject.getClusterGraph(clusterOutputData)
    clusterPlot.update_layout(title="Output Clusters")
    cluster_data = []
    ClusterMean = []
    boxPlotData = []
    color = "White"
    if n_clicks1 > cluster_click1:
        cluster_click1 = n_clicks1
        cluster_data = clusterOutputData[0]
        color = "DarkOrange"
        ClusterMean = clusterMean[0]
        boxPlotData = boxplotDataframe[0]


    if n_clicks2 > cluster_click2:
        cluster_click2 = n_clicks2
        cluster_data = clusterOutputData[1]
        color = "Crimson"
        ClusterMean = clusterMean[1]
        boxPlotData = boxplotDataframe[1]

    if n_clicks3 > cluster_click3:
        cluster_click3 = n_clicks3
        cluster_data = clusterOutputData[2]
        color = "RebeccaPurple"
        ClusterMean = clusterMean[2]
        boxPlotData = boxplotDataframe[2]

    if n_clicks4 > cluster_click4:
        cluster_click4 = n_clicks4
        cluster_data = clusterOutputData[3]
        color = "mediumvioletred"
        ClusterMean = clusterMean[3]
        boxPlotData = boxplotDataframe[3]

    if n_clicks5 > cluster_click5:
        cluster_click5 = n_clicks5
        cluster_data = clusterOutputData[4]
        color = "pink"
        ClusterMean = clusterMean[4]
        boxPlotData = boxplotDataframe[4]


    if len(boxPlotData) != 0:
        # ClusterMean.pop()
        boxPlot = px.box(y=boxPlotData.Sigma_1, title="Sigma_1")
        boxPlot1 = px.box(y=boxPlotData.Sigma_2, title="Sigma_2")
        boxPlot2 = px.box(y=boxPlotData.A, title="A")
        boxPlot3 = px.box(y=boxPlotData.BeltSpinRatio, title="BeltSpinRatio")
        boxPlot4 = px.box(y=boxPlotData.SpinPositionsPerMeterInverse, title="SpinPositionsPerMeter")

    if len(ClusterMean) != 0:
        # ClusterMean.pop()
        meanPlot =  px.bar(x=["Half","One","Two","Five","Ten","Twenty","Fifty"], y=ClusterMean[:-1], title="Mean CV Value Vs grid-size")
        meanPlot.update_yaxes(range=[0, 120])


    if len(cluster_data) != 0:
        clusterPlot.update_layout(
        shapes=[
            # unfilled circle
            dict(
                type="circle",
                xref="x",
                yref="y",
                x0=min(cluster_data.pc_1), y0=min(cluster_data.pc_2),
                x1=max(cluster_data.pc_1), y1=max(cluster_data.pc_2),
                line_color=color,
            ),
        ])

    return clusterPlot, meanPlot, boxPlot, boxPlot1, boxPlot2, boxPlot3, boxPlot4

@app.callback(dash.dependencies.Output('DLS1', 'value'),
	[dash.dependencies.Input('DS1','value')])
def update_CVResGraph(value):
    global Sigma1
    Sigma1 = value
    return value

@app.callback(dash.dependencies.Output('DLS2', 'value'),
	[dash.dependencies.Input('DS2','value')])
def update_CVResGraph(value):
    global Sigma2
    Sigma2 = value
    return value

@app.callback(dash.dependencies.Output('DLA', 'value'),
	[dash.dependencies.Input('DA','value')])
def update_CVResGraph(value):
    global A
    A = value
    return value

@app.callback(dash.dependencies.Output('DLBSR', 'value'),
	[dash.dependencies.Input('DBSR','value')])
def update_CVResGraph(value):
    global BSR
    BSR = value
    return value

@app.callback(dash.dependencies.Output('DLSPM', 'value'),
	[dash.dependencies.Input('DSPM','value')])
def update_CVResGraph(value):
    global SPM
    SPM = value
    return value

@app.callback([dash.dependencies.Output('DS1', 'style'),
    dash.dependencies.Output('DS2', 'style'),
    dash.dependencies.Output('DA', 'style'),
    dash.dependencies.Output('DBSR', 'style'),
    dash.dependencies.Output('DSPM', 'style')],
	[dash.dependencies.Input('DInputs','value')])
def highlightDerivativeLabel(value):
    if value =='DS1':
        return {"box-shadow": "5px 10px 1px black"},{},{},{},{}
    if value == 'DS2':
        return  {},{"box-shadow": "5px 10px 1px black"},{},{},{}
    if value == 'DA':
        return {},{},{"box-shadow": "5px 10px 1px black"},{},{}
    if value == 'DBSR':
        return {},{},{},{"box-shadow": "5px 10px 1px black"},{}
    if value == 'DSPM':
        return {},{},{},{},{"box-shadow": "5px 10px 1px black"}
    return {},{},{},{},{}

    
@app.callback(dash.dependencies.Output('DerivativePlot', 'figure'),
    [dash.dependencies.Input('DS1', 'value'),
    dash.dependencies.Input('DS2', 'value'),
    dash.dependencies.Input('DA', 'value'),
    dash.dependencies.Input('DBSR', 'value'),
    dash.dependencies.Input('DSPM', 'value'),
	dash.dependencies.Input('DInputs','value'),
    dash.dependencies.Input('dresolution','value')])
def updatDerivativePlot(value1, value2, value3, value4, value5, value6, res):
    global sigma1_derivative_cur, sigma2_derivative_cur, a_derivative_cur, bsr_derivative_cur, spm_derivative_cur, derivData, derivPlot, frozen_deriv_ip, DResolution
    if res is not None:
        DResolution = res

    if value1 != sigma1_derivative_cur:
        sigma1_derivative_cur = value1

    if value2 != sigma2_derivative_cur:
        sigma2_derivative_cur = value2

    if value3 != a_derivative_cur:
        a_derivative_cur = value3

    if value4 != bsr_derivative_cur:
        bsr_derivative_cur = value4

    if value5 != spm_derivative_cur:
        spm_derivative_cur = value5

    if value6 =='DS1':
        frozen_deriv_ip = "Sigma_1"
        derivData = DataPrepUtils.getDerivativeDataFrame("Sigma_1", sigma2_derivative_cur, a_derivative_cur, bsr_derivative_cur, spm_derivative_cur, DResolution, 100)

    if value6 == 'DS2':
        frozen_deriv_ip = "Sigma_2"
        derivData = DataPrepUtils.getDerivativeDataFrame("Sigma_2", sigma1_derivative_cur, a_derivative_cur, bsr_derivative_cur, spm_derivative_cur, DResolution, 100)

    if value6 == 'DA':
        frozen_deriv_ip = "A"
        derivData = DataPrepUtils.getDerivativeDataFrame("A", sigma1_derivative_cur, sigma2_derivative_cur, bsr_derivative_cur, spm_derivative_cur, DResolution, 100)

    if value6 == 'DBSR':
        frozen_deriv_ip = "BeltSpinRatio"
        derivData = DataPrepUtils.getDerivativeDataFrame("BeltSpinRatio", sigma1_derivative_cur, sigma2_derivative_cur, a_derivative_cur, spm_derivative_cur, DResolution, 100)

    if value6 == 'DSPM':
        frozen_deriv_ip = "SpinPositionsPerMeterInverse"
        derivData = DataPrepUtils.getDerivativeDataFrame("SpinPositionsPerMeterInverse", sigma1_derivative_cur, sigma2_derivative_cur, a_derivative_cur, bsr_derivative_cur, DResolution, 100)

    if DResolution == 7:
        derivDataAll = DataPrepUtils.getDerivativeDataFrame(frozen_deriv_ip, sigma1_derivative_cur, sigma2_derivative_cur, a_derivative_cur, bsr_derivative_cur, DResolution, 100)
        derivPlotAll = go.Figure()
        derivPlotAll.add_trace(go.Scatter(x=derivDataAll[frozen_deriv_ip], y=derivDataAll['Derivative_half'], name="Res: 0.5"))
        derivPlotAll.add_trace(go.Scatter(x=derivDataAll[frozen_deriv_ip], y=derivDataAll['Derivative_1'], name="Res: 1"))
        derivPlotAll.add_trace(go.Scatter(x=derivDataAll[frozen_deriv_ip], y=derivDataAll['Derivative_2'], name="Res: 2"))
        derivPlotAll.add_trace(go.Scatter(x=derivDataAll[frozen_deriv_ip], y=derivDataAll['Derivative_5'], name="Res: 5"))
        derivPlotAll.add_trace(go.Scatter(x=derivDataAll[frozen_deriv_ip], y=derivDataAll['Derivative_10'], name="Res: 10"))
        derivPlotAll.add_trace(go.Scatter(x=derivDataAll[frozen_deriv_ip], y=derivDataAll['Derivative_20'], name="Res: 20"))
        derivPlotAll.add_trace(go.Scatter(x=derivDataAll[frozen_deriv_ip], y=derivDataAll['Derivative_50'], name="Res: 50"))

        derivPlotAll.update_layout(
            title="Partial Derivative Plot",
            xaxis_title=frozen_deriv_ip,
            yaxis_title="Derivative",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            ),
            xaxis=dict(rangeselector=dict(visible=True), rangeslider=dict(visible=True))
        )

        derivPlotAll.update_layout(
            width=900,
            height=700,
            autosize=False,
            margin=dict(t=0, b=0, l=0, r=0),
            template="plotly_white",
        )

        return derivPlotAll

    derivPlot = go.Figure(data=go.Scatter(x=derivData[frozen_deriv_ip], y=derivData['Derivative']))
    derivPlot.update_layout(
        title="Partial Derivative Plot",
        xaxis_title=frozen_deriv_ip,
        yaxis_title="Derivative",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        ),
        xaxis=dict(rangeselector=dict(visible=True), rangeslider=dict(visible=True))
    )

    derivPlot.update_layout(
            width=900,
            height=700,
            autosize=False,
            margin=dict(t=0, b=0, l=0, r=0),
            template="plotly_white",
        )
    
    return derivPlot

if __name__ == '__main__':
	return app.run_server(debug=True)
