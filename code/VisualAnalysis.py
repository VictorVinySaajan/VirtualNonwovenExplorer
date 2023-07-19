import numpy as np
import pandas as pd
import plotly.graph_objects as go

class FibreGraphs():
    def getCVtoResolutionGraph(self, predictions):
        resolutions = [0.5,1,2,5,10,20,50]
        fig = go.Figure(data=go.Scatter(x=resolutions, y=predictions))
        fig.update_layout(
            title="CV Values at different grid-sizes",
            xaxis_title="grid-sizes(mm)",
            yaxis_title="CV Values",
            font=dict(
                family="calibri",
                size=18,
                color="#7f7f7f"
            )
        )
        return fig

    def getCliGraph(self, predictions):
        fig = go.Figure(data=go.Bar(x=[1], y=[predictions], width=0.1))
        fig.update_layout(
            title="Cloudiness Index",
            xaxis_title="",
            yaxis_title="Cli Values",
            font=dict(
                family="calibri",
                size=18,
                color="#7f7f7f"
            ),
            xaxis=dict(showticklabels=False)
        )
        return fig

    def getCVtoResolutionComparisionGraph(self, predictions):
        resolutions = [0.5,1,2,5,10,20,50]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=resolutions, y=predictions[0],
                            mode='lines',
                            name='ds = 2.5e-5'))
        fig.add_trace(go.Scatter(x=resolutions, y=predictions[1],
                            mode='lines',
                            name='ds = 5e-5'))

        fig.update_layout(
            title="CV Values at different grid-sizes",
            xaxis_title="grid-sizes(mm)",
            yaxis_title="CV Values",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
        return fig

    def getCVtoResolutionGraphForComparision(self, sampleList):
        fig = go.Figure()
        resolutions = [0.5,1,2,5,10,20,50]
        for i in sampleList:
            fig.add_trace(go.Scatter(x=resolutions, y=i))
        fig.update_layout(
            title="CV Values at different grid-sizes",
            xaxis_title="grid-sizes(mm)",
            yaxis_title="CV Values",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
        return fig

    def getCliGraphForComparision(self, sampleList):
        fig = go.Figure()
        j = 1
        for i in sampleList:
            fig.add_trace(go.Bar(x=[j], y=[i], width=0.1))
            j+=1
        fig.update_layout(
            title="Cloudiness Index",
            xaxis_title="",
            yaxis_title="Cli Values",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            ),
            xaxis=dict(showticklabels=False)
        )
        return fig

    def getEmptyPlot(self):
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
            plot_bgcolor="white"
        )
        return fig

    def get3DSurfacePlot(self, inputName, dataframe, res):
        # print("dataframe", dataframe.values)
        # print("np.array(dataframe.index)", np.array(dataframe.index))
        fig = go.Figure(data=[go.Surface(z=dataframe.values.tolist(), y = np.array(dataframe.index), x = np.array(dataframe.columns),
        colorscale="Viridis", 
        hovertemplate =
        '<br><b>Input_Value2</b>: %{x}<br>'+
        '<br><b>Input_Value1</b>: %{y}<br>'+
        '<br><b>CV_Value</b>: %{z}<br>',
        showlegend = False)])

        if res != 7:
            fig.update_layout(title="3D Surface plot for the whole range of {}".format(inputName), autosize=False,
                    width=1000, height=1000,
                    )
            fig.update_layout(scene = dict(
                        xaxis_title='{}'.format(inputName.split('and')[1]),
                        yaxis_title='{}'.format(inputName.split('and')[0]),
                        zaxis_title='CV VALUES'),
                        font=dict(family="Courier New, monospace", size=14))
        else:
            fig.update_layout(title="3D Surface plot for the whole range of {}".format(inputName), autosize=False,
                    width=1000, height=1000,
                    )
            fig.update_layout(scene = dict(
                        xaxis_title='{}'.format(inputName.split('and')[1]),
                        yaxis_title='{}'.format(inputName.split('and')[0]),
                        zaxis_title='Cloudiness Index'),
                        font=dict(family="Courier New, monospace", size=14))
        return fig

    def getHeatMap(self, inputName, dataframe):
        fig = go.Figure(data=[go.Heatmap(z=dataframe.values, x = np.array([0.5,1,2,5,10,20,50]), y = np.array(dataframe.index),
        colorscale="Viridis",
        hovertemplate =
        '<br><b>grid-size</b>: %{x}<br>'+
        '<br><b>CV_Value</b>: %{z}<br>'+
        '<br><b>Input Value</b>: %{y}<br>',
        # '<b>%{text}</b>',
        # text = [['Input Value {}'.format(dataframe.index[i]) for j in range(dataframe.shape[1])] for i in range(dataframe.shape[0])],
        showlegend = False)])

        fig.update_layout(title="Heatmap for the whole range of {}".format(inputName))
        return fig
        
    def getClusterGraph(self, clusterList):
        fig = go.Figure()
        color = ["DarkOrange","Crimson","RebeccaPurple","mediumvioletred","pink","mediumorchid"]
        label = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"]

        for i in range(0,5):
            fig.add_trace(
                go.Scatter(
                    x=clusterList[i].pc_1,
                    y=clusterList[i].pc_2,
                    mode="markers",
                    marker=dict(color=color[i]),
                    name=label[i]
                )
            )
    
        return fig

    def getCVForIndividualResolutions(self, predictions, cli_predictions, dataframe):
        trace_names = ["CV grid-size : 0.5 mm", "CV grid-size : 1mm", "CV grid-size : 2mm", "CV grid-size : 5mm", "CV grid-size : 10mm", "CV grid-size : 20mm", "CV grid-size : 50mm"]
        fig = go.Figure()
        #print("cli predictions", len(cli_predictions))

        for i in range(0, len(predictions)):
            if i ==0:
                for j in range(0,7):
                    fig.add_trace(
                    go.Scatter(x=list(dataframe[i]["Sigma_1"]), y=predictions[i][:,j], visible = "legendonly", name= trace_names[j]))
                fig.add_trace(
                go.Scatter(x=list(dataframe[i]["Sigma_1"]), y=cli_predictions[i], visible = True, name= "Cloudiness Index"))

            if i ==1:
                for j in range(0,7):
                    fig.add_trace(
                    go.Scatter(x=list(dataframe[i]["Sigma_2"]), y=predictions[i][:,j], visible = False, name= trace_names[j]))
                fig.add_trace(
                go.Scatter(x=list(dataframe[i]["Sigma_2"]), y=cli_predictions[i], visible = False, name= "Cloudiness Index"))

            if i ==2:
                for j in range(0,7):
                    fig.add_trace(
                    go.Scatter(x=list(dataframe[i]["A"]), y=predictions[i][:,j], visible = False, name= trace_names[j]))
                fig.add_trace(
                go.Scatter(x=list(dataframe[i]["A"]), y=cli_predictions[i], visible = False, name= "Cloudiness Index"))

            if i ==3:
                for j in range(0,7):
                    fig.add_trace(
                    go.Scatter(x=list(dataframe[i]["BeltSpinRatio"]), y=predictions[i][:,j], visible = False, name= trace_names[j]))
                fig.add_trace(
                go.Scatter(x=list(dataframe[i]["BeltSpinRatio"]), y=cli_predictions[i], visible = False, name= "Cloudiness Index"))

            if i ==4:
                for j in range(0,7):
                    fig.add_trace(
                    go.Scatter(x=list(dataframe[i]["SpinPositionsPerMeterInverse"]), y=predictions[i][:,j], visible = False, name= trace_names[j]))
                fig.add_trace(
                go.Scatter(x=list(dataframe[i]["SpinPositionsPerMeterInverse"]), y=cli_predictions[i], visible = False, name= "Cloudiness Index"))

        # visible= "legendonly", hide tarce while displaying lengends which avn be toggled
        fig.update_layout(
            updatemenus=[
                dict(
                    type = "buttons",
                    direction = "left",
                    buttons=list([
                        dict(
                            args=[{"visible": ["legendonly","legendonly","legendonly","legendonly","legendonly","legendonly","legendonly",True,
                                                False,False,False,False,False,False,False,False,
                                                False,False,False,False,False,False,False,False,
                                                False,False,False,False,False,False,False,False,
                                                False,False,False,False,False,False,False,False]}],
                            label="Sigma_1",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [False,False,False,False,False,False,False,False,
                                                "legendonly","legendonly","legendonly","legendonly","legendonly","legendonly","legendonly",True,
                                                False,False,False,False,False,False,False,False,
                                                False,False,False,False,False,False,False,False,
                                                False,False,False,False,False,False,False,False]}],
                            label="Sigma_2",
                            method="restyle"
                        ),
                        dict(
                             args=[{"visible": [False,False,False,False,False,False,False,False,
                                                False,False,False,False,False,False,False,False,
                                                "legendonly","legendonly","legendonly","legendonly","legendonly","legendonly","legendonly",True,
                                                False,False,False,False,False,False,False,False,
                                                False,False,False,False,False,False,False,False]}],
                            label="A",
                            method="restyle"
                        ),
                        dict(
                             args=[{"visible": [False,False,False,False,False,False,False,False,
                                                False,False,False,False,False,False,False,False,
                                                False,False,False,False,False,False,False,False,
                                                "legendonly","legendonly","legendonly","legendonly","legendonly","legendonly","legendonly",True,
                                                False,False,False,False,False,False,False,False]}],
                            label="BeltSpinRatio",
                            method="restyle"
                        ),
                        dict(
                             args=[{"visible": [False,False,False,False,False,False,False,False,
                                                False,False,False,False,False,False,False,False,
                                                False,False,False,False,False,False,False,False,
                                                False,False,False,False,False,False,False,False,
                                                "legendonly","legendonly","legendonly","legendonly","legendonly","legendonly","legendonly",True]}],
                            label="SpinPositionsPerMeter",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.15,
                    xanchor="left",
                    y=1.2,
                    yanchor="top"
                ),
            ]
        )

        fig.update_layout(
            title_text="Individual grid-sizes",
			yaxis_title="Values",
            annotations=[
                dict(text="Input Features:", showarrow=False,
                                    x=0, y=1.08, yref="paper", align="left")
            ],
            legend=dict(font=dict(size=20))
        )

        return fig

            
        
        
