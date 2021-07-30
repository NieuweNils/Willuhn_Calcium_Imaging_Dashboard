import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import flask
from flask import Flask, Response

import numpy as np
import plotly.graph_objects as go
from scipy.io import loadmat
import skimage.io as skim_io
# import skvideo.io as skvid_io

server = Flask(__name__)

# Data processing
# TODO: put in separate file

# For position
data_after_CNMF_E = loadmat(""../data/concat_may_NoRMCorre_results.mat"")
data_after_CNMF_E = data_after_CNMF_E["results"]

# For animated line plot of Fluorescence trace
fluorescence_traces = np.array(data_after_CNMF_E["C"][0][0])
layout = {"title": "Neuron " + str(7)}
trace = fluorescence_traces[7, 0:2500]

# For video of Fluorescence traces
# tiff_data = skim_io.imread("../data/concat_may.tif")


def animated_heatmap(data, layout={}):
    figure_settings = {"data": [],
                       "layout": layout,
                       # "frames": []
                       }

    # n_images = (data.shape[0])
    # frames = []
    # for frame in range(0, n_images, 1000):
    #     image = data[frame]
    #     curr_frame = go.Frame(data=[go.Heatmap(z=image, colorscale="gray")])
    #     frames.append(curr_frame)

    figure_settings["layout"]["xaxis"] = {"range": [0, data.shape[0]]}
    figure_settings["layout"]["yaxis"] = {"range": [0, data.shape[1]]}

    figure_settings["data"] = [go.Heatmap(z=data, colorscale="gray")]
    figure_settings["layout"]["updatemenus"] = play_and_pause_buttons()
    # figure_settings["frames"] = frames
    return go.Figure(figure_settings)


# Dashboard setup
def play_and_pause_buttons():
    layout = [{
        "type": "buttons",
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 50, "redraw": False},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 0}}]
            },
            {
                "label": "Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}]
            }
        ]
    }]
    return layout


def animated_linechart(data, layout={}):
    figure_settings = {"data": [],
                       "layout": layout,
                       "frames": []}
    x_axis = list(range(data.shape[0]))
    y_axis = list(data)
    frames = []
    for frame in range(0, len(x_axis), 50):
        x_axis_frame = np.arange(frame)
        y_axis_frame = data[0:frame]
        curr_frame = go.Frame(data=[go.Scatter(x=x_axis_frame, y=y_axis_frame, mode="lines")])
        frames.append(curr_frame)

    figure_settings["layout"]["xaxis"] = {"title": "time (ms)",
                                          "range": [0, len(x_axis)]}
    figure_settings["layout"]["yaxis"] = {"title": "intensity of signal (value??)",
                                          "range": [0, max(y_axis)]}
    figure_settings["data"] = [go.Scatter(x=x_axis, y=y_axis, mode="lines")]
    figure_settings["layout"]["updatemenus"] = play_and_pause_buttons()
    figure_settings["frames"] = frames
    return go.Figure(figure_settings)


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash("name", external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Ca-imaging dashboard'),

    dcc.Upload(html.Button('Upload File')),

    html.Video(src="../jupyter/output.webm", controls=True),

    dcc.Graph(
        id='example-line-chart',
        figure=animated_linechart(data=trace, layout=layout)
    ),

    # dcc.Graph(
    #     id='example-heatmap',
    #     figure=animated_heatmap(data=tiff_data[0])
    # ),
])


@server.route('/static/<path:path>')
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(os.path.join(root_dir, 'static'), path)


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)
