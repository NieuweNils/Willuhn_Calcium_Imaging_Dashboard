from copy import copy

import numpy as np
import plotly.graph_objs as go
from formatting import layout as standard_layout

from data_processing import retrieve_metadata, get_pixel_df, get_cols_and_rows


def cell_outline_chart(locations_df, metadata, cell_number):
    """
    :param locations_df: a pandas dataframe with locations of each neuron, as stored in the the matlab variable "A" in the output of the CNMF_E algorithm
    :param metadata: a dictionary with all "options" variables, retrieved from  the the matlab variable "options" in the output of the CNMF_E algorithm
    :param cell_number: an integer defining the cell which outline is to be displayed
    :return: a plotly.Figure object containing a plotly.graph_obj.Heatmap displaying the location of the cell of interest
    """
    d1 = metadata['d1']  # width of recording in #pixels (or was it height?)
    d2 = metadata['d2']  # height of recording in #pixels (or was it width?)
    pixel_df = get_pixel_df(locations_df)
    cols, rows = get_cols_and_rows(pixel_df, metadata)
    neuron_positions = np.stack((rows, cols))

    white_background = np.zeros((d1, d2))
    neuron_position = neuron_positions[:, cell_number, :]
    neuron_position = neuron_position[~np.isnan(neuron_position)]
    amount_of_pixels = int(neuron_position.shape[0] / 2)
    neuron_position = np.reshape(neuron_position, (2, amount_of_pixels))

    image_neuron = copy(white_background)
    for j in range(amount_of_pixels):
        row = int(neuron_position[0, j])
        col = int(neuron_position[1, j])
        image_neuron[row, col] = 1

    heatmap = go.Heatmap(z=image_neuron,
                         colorscale='gray')
    layout = {'title': f'contour of cell {cell_number}'}
    figure = go.Figure(data=heatmap,
                       layout=layout)

    return figure


def generatorify(tiff_data, skip_rate=1):
    """
    :param tiff_data: a 3D numpy array of dimensions [frames, height, width] that stores grayscale images
    :param skip_rate:  only every {this number} frames in the
    np.array are used for the animation (default: every frame)
    :return: a generator object that yields each image of the video per yield
    """
    count = 0
    for image in tiff_data:
        if count % skip_rate == 0:
            yield image
        count += 1


def play_and_pause_buttons(duration_play=50,
                           duration_pause=0):
    """
    :param duration_play: amount (in ms) that each frame stays on the screen
    :param duration_pause: amount (in ms) before the pause of the animation kicks in
    :return: a 'layout' dict that can be used as a plotly.graph_objs.figure['layout'] field
    """
    layout = [{
        "type": "buttons",
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": duration_play, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 0}}]
            },
            {
                "label": "Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": duration_pause, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}]
            }
        ]
    }]
    return layout


def animated_line_chart(data, layout=standard_layout):
    """
    :param data: a 2d numpy array of dimensions [channel, timestamp] that stores fluorescence traces for each neuron
    :param layout: settings to use in plotly.graph_objs.figure['layout']
    :return: an animated linechart object (of
    type plotly.graph_objs.figure where keys ['data'] and ['frames'] store plotly.graph_objs.Scatter objects)
    """
    figure_settings = {
        "data": [],
        "layout": layout,
        "frames": []
    }
    x_axis = list(range(data.shape[0]))
    y_axis = list(data)
    frames = []
    for frame in range(0, len(x_axis), 50):
        x_axis_frame = np.arange(frame)
        y_axis_frame = data[0:frame]
        curr_frame = go.Frame(data=[go.Scatter(x=x_axis_frame, y=y_axis_frame, mode="lines")])
        frames.append(curr_frame)

    figure_settings["layout"]["xaxis"]["title"] = "time (ms)"
    figure_settings["layout"]["xaxis"]["range"] = [0, len(x_axis)]
    figure_settings["layout"]["yaxis"]["title"] = "intensity of signal (value??)"
    figure_settings["layout"]["yaxis"]["range"] = [0, max(y_axis)]
    figure_settings["data"] = [go.Scatter(x=x_axis, y=y_axis, mode="lines")]
    figure_settings["layout"]["updatemenus"] = play_and_pause_buttons()
    figure_settings["frames"] = frames

    return go.Figure(figure_settings)


def animated_heatmap(data, layout=standard_layout, skip_rate=1):
    """
    :param data: a 3D numpy array of dimensions [frames, height, width] that stores grayscale images
    :param layout:
    settings to use in plotly.graph_objs.figure['layout']
    :param skip_rate: only every {this number} frames in the
    np.array are used for the animation (default: every frame)
    :return: an animated heatmap object (of type
    plotly.graph_objs.figure where keys ['data'] and ['frames'] store plotly.graph_objs.Heatmap objects)
    """
    figure_settings = {
        "data": [],
        "layout": layout,
        "frames": []}

    figure_settings["layout"]["xaxis"] = {"range": [0, data.shape[1]]}
    figure_settings["layout"]["yaxis"] = {"range": [0, data.shape[2]]}

    frames = []
    for image in generatorify(data, skip_rate=skip_rate):
        curr_frame = go.Frame(data=[go.Heatmap(z=image, colorscale='gray')])
        frames.append(curr_frame)

    figure_settings["data"] = [go.Heatmap(z=data[0], colorscale='gray')]
    figure_settings["layout"]["updatemenus"] = play_and_pause_buttons(duration_play=400)
    figure_settings["frames"] = frames

    heatmap = go.Figure(figure_settings)

    return heatmap
