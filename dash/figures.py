from copy import copy

import numpy as np
import plotly.graph_objs as go

from data_processing import get_pixel_df, get_cols_and_rows
from formatting import layout as standard_layout


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


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }


def slider_base():
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Cell:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 1,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    return sliders_dict


def slider_steps(frame_names):
    steps = [
        {
            "args": [[name], frame_args(0)],
            "label": str(index),
            "method": "animate",
        }
        for index, name in enumerate(frame_names)
    ]

    return steps


def drop_down(frame_names):
    drop_down_dict = {
        'buttons' :
        [
            {"args": [[name], frame_args(0)],
             'label': f'Cell {name}',
             'method': "animate",} for name in frame_names
        ],
        'direction': 'down',
        'pad': {'r': 10, 't':10},
        'showactive': True,
        'x': 0,
        'xanchor': 'left',
        'y': 1.5,
        'yanchor': 'top'
    }
    return drop_down_dict


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


def transform_data_cell_outline_plot(locations_df, metadata):
    d1 = metadata['d1']  # width of recording in #pixels (or was it height?)
    d2 = metadata['d2']  # height of recording in #pixels (or was it width?)
    pixel_df = get_pixel_df(locations_df)
    cols, rows = get_cols_and_rows(pixel_df, metadata)
    neuron_positions = np.stack((rows, cols))
    number_of_cells = neuron_positions.shape[1]

    return neuron_positions, number_of_cells, d1, d2


def create_frames_cell_outline_plot(number_of_cells, neuron_positions, d1, d2):
    white_background = np.zeros((d1, d2))
    frames = []
    frame_names = []
    for cell_number in range(number_of_cells):
        neuron_position = neuron_positions[:, cell_number, :]
        neuron_position = neuron_position[~np.isnan(neuron_position)]
        amount_of_pixels = int(neuron_position.shape[0] / 2)
        neuron_position = np.reshape(neuron_position, (2, amount_of_pixels))
        image_neuron = copy(white_background)
        for j in range(amount_of_pixels):
            row = int(neuron_position[0, j])
            col = int(neuron_position[1, j])
            image_neuron[row, col] = 1
        frame_name = str(cell_number)
        curr_frame = go.Frame(data=[go.Heatmap(z=image_neuron, colorscale='gray')],
                              name=frame_name)  # IT'S VERY IMPORTANT TO NAME EACH FRAME THE SAME AS EACH SLIDER STEP
                                                # & DROP DOWN!!
        frames.append(curr_frame)
        frame_names.append(frame_name)
    return frames, frame_names


# TODO: CHECK IF THE ANIMATION WORKS, ADD SLIDER, AND CREATE OVERLAYS OF MULTIPLE NEURONS
def cell_outlines(locations_df, metadata):
    """
    :param locations_df: a pandas dataframe with locations of each neuron, as stored in the the matlab variable "A" in the output of the CNMF_E algorithm
    :param metadata: a dictionary with all "options" variables, retrieved from  the the matlab variable "options" in the output of the CNMF_E algorithm
    :return: a plotly.Figure object containing plotly.graph_obj.Heatmap objects displaying the location of the neurons
    """
    # create frames
    neuron_positions, number_of_cells, d1, d2 = transform_data_cell_outline_plot(locations_df, metadata)
    frames, frame_names = create_frames_cell_outline_plot(number_of_cells, neuron_positions, d1, d2)

    # create slider & drop down
    slider_dict = slider_base()
    slider_dict["steps"] = slider_steps(frame_names)
    drop_down_settings = drop_down(frame_names)

    # Assemble figure
    first_heatmap = frames[0]['data']
    layout = {
        'title': {'text': f'position of cell',
                  'x': 0.5},
        'font': {'size': 18},
        "sliders": [slider_dict],
        "updatemenus": [drop_down_settings]
    }
    figure = go.Figure(data=first_heatmap,
                       layout=layout,
                       frames=frames)

    return figure
