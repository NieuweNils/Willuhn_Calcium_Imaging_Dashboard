import base64
import io
from copy import copy

import numpy as np
import plotly.graph_objs as go
from PIL import Image

from data_processing import retrieve_contour_coordinates
from formatting import standard_layout


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
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 1,
        "x": 0.2,
        "y": 0.1,
        "steps": []
    }

    return sliders_dict


def slider_steps(frame_names):
    steps = [
        {
            "args": [[name], frame_args(0)],
            "label": name,
            "method": "animate", } for name in frame_names
    ]

    return steps


def drop_down(frame_names):
    drop_down_dict = {
        'buttons':
            [
                {"args": [[name], frame_args(0)],
                 'label': f'{name[:-13]}', # take off the " + neighbours" part of the name
                 'method': "animate", } for name in frame_names
            ],
        'direction': 'up',
        'pad': {'b': 0, 't': 50},
        'showactive': True,
        'x': 0.1,
        'y': 0,
    }
    return drop_down_dict


def animated_line_chart(data, layout_base=standard_layout):
    """
    :param data: a 2d numpy array of dimensions [channel, timestamp] that stores fluorescence traces for each neuron
    :param layout_base: settings to use in plotly.graph_objs.figure['layout']
    :return: an animated line chart object (of
    type plotly.graph_objs.figure where keys ['data'] and ['frames'] store plotly.graph_objs.Scatter objects)
    """

    layout = copy(layout_base)  # Copy by value instead of reference
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


def animated_heatmap(data, layout_base=standard_layout, skip_rate=1):
    """
    :param data: a 3D numpy array of dimensions [frames, height, width] that stores grayscale images
    :param layout_base:
    settings to use in plotly.graph_objs.figure['layout']
    :param skip_rate: only every {this number} frames in the
    np.array are used for the animation (default: every frame)
    :return: an animated heatmap object (of type
    plotly.graph_objs.figure where keys ['data'] and ['frames'] store plotly.graph_objs.Heatmap objects)
    """
    layout = copy(layout_base)  # Copy by value instead of reference
    figure_settings = {
        "data": [],
        "layout": layout,
        "frames": []}

    figure_settings["layout"]["xaxis"]["range"] = [0, data.shape[2]]
    figure_settings["layout"]["yaxis"]["range"] = [0, data.shape[1]]

    frames = []
    for image in generatorify(data, skip_rate=skip_rate):
        curr_frame = go.Frame(data=[go.Heatmap(z=image, colorscale='gray')])
        frames.append(curr_frame)

    figure_settings["data"] = [go.Heatmap(z=data[0], colorscale='gray')]
    figure_settings["layout"]["updatemenus"] = play_and_pause_buttons(duration_play=400)
    figure_settings["frames"] = frames

    heatmap = go.Figure(figure_settings)

    return heatmap


def layout_cell_outline_plot(layout, frame_names, background, d1, d2):
    slider_dict = slider_base()
    slider_dict["steps"] = slider_steps(frame_names)
    layout["sliders"] = [slider_dict]

    drop_down_settings = drop_down(frame_names)
    layout["updatemenus"] = [drop_down_settings]

    # convert background to image
    background_pil = Image.fromarray(np.uint8(np.array(background)*256))
    buffer = io.BytesIO()
    background_pil.save(buffer, format="jpeg")
    background_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    layout["xaxis"]["range"] = [0, d2]
    layout["yaxis"]["range"] = [d1, 0]
    # place background image to fill screen
    layout["images"] = [
        {"source": "data:image/jpeg;base64,{}".format(background_base64),
         "xref": "paper",
         "yref": "paper",
         "x": 0,
         "y": 1,
         "sizex": 1,
         "sizey": 1,
         "xanchor": "left",
         "yanchor": "top",
         "sizing": "stretch",
         "layer": "below"}
    ]

    return layout


def correlation_plot(cell_list, correlation_df, distances, min_correlation=0.2, max_distance=10, layout_base=standard_layout):
    # Discard correlation values below threshold
    highly_correlating_neurons = correlation_df[correlation_df > min_correlation]
    # Discard non-neighbours
    neighbours = distances[distances[:, 2] < max_distance]
    cells_to_select_row = set(neighbours[:, 0])
    cells_to_select_col = set(neighbours[:, 1])
    correlating_neighbours = highly_correlating_neurons.loc[cells_to_select_row][cells_to_select_col]
    correlating_neighbours = correlating_neighbours.dropna(how="all").dropna(how="all", axis=1)
    # TODO: add bit here that discards correlations of neurons that are not pairs in distance_df
    # Assemble the figure
    values = correlating_neighbours.values
    values = values[~np.isnan(correlating_neighbours)]
    middle = np.average(values)
    maximum = np.max(values)
    cols = [f"cell {column}" for column in correlating_neighbours.columns]
    rows = [f"cell {index}" for index in correlating_neighbours.index]
    layout = copy(layout_base)
    layout["title"] = "Correlating neighbours"

    figure = go.Figure(data=[go.Heatmap(z=correlating_neighbours,
                                        x=cols,
                                        y=rows,
                                        zmax=maximum,
                                        zmid=middle,
                                        zmin=0,
                                        colorscale="ice"
                                        )],
                       layout=layout)

    return figure


def line_chart(cells_to_display, traces, layout_base=standard_layout):
    """
    :param traces: a 2d numpy array of dimensions [channel, timestamp] that stores fluorescence traces for each neuron
    :param layout_base: settings to use in plotly.graph_objs.figure['layout']
    :return: a line chart object (of type plotly.graph_objs.figure where keys ['traces'] stores plotly.graph_objs.Scatter
     objects)
    """
    figure = go.Figure()
    stop = len(traces[str(cells_to_display[0])])  # stop == the amount of time points in the first dictionary entry
    step = int(stop / 10000)  # reduce the data load a bit, use max 10,000 time points
    x_axis = list(range(0, stop, step))
    cells = traces.keys()
    cells_to_display = sorted(cells_to_display)

    traces_to_display = [(cell, traces[cell]) for cell in cells if int(cell) in cells_to_display]
    for (cell, trace) in traces_to_display:
        figure.add_trace(go.Scatter(x=x_axis, y=trace[0:stop:step], name="cell " + str(cell), mode="lines"))

    layout = copy(layout_base)  # Copy by value, NOT by reference
    layout["xaxis"]["title"] = "time (ms?)"
    layout["xaxis"]["range"] = [0, x_axis[-1]]
    layout["yaxis"]["title"] = "intensity of signal (C_raw)"
    layout["yaxis"]["range"] = [np.min([scatter["y"] for scatter in figure["data"]]),
                                np.max([scatter["y"] for scatter in figure["data"]])]
    layout["autosize"] = False

    figure.layout = layout

    return figure


def contour_plot(loc_dict, background, cells_per_row, layout_base=standard_layout):
    d1, d2 = (len(background), len(background[0]))
    layout_base = copy(layout_base)  # Copy by value instead of reference

    #  extract coordinates
    locations = np.array([np.array(lst) for lst in loc_dict.values()]).T
    contour_matrix = retrieve_contour_coordinates(locations=locations, background=background)
    contour_coordinates = {}
    i = 0
    for cell in loc_dict:
        contour_coordinates[cell] = contour_matrix[i]
        i += 1

    # create frames
    frames = []
    for row in cells_per_row:
        contour_trace = []
        for cell in row:  # loop through all cells & NaN values
            if np.isnan(cell):  # fix for it showing old traces in a new frame
                contour_trace.append(go.Scatter(x=[],
                                                y=[],
                                                mode="lines",
                                                name="NaN"))
            else:
                coordinate_vector = contour_coordinates[str(int(cell))]
                contour_trace.append(go.Scatter(x=coordinate_vector[:, 0],
                                                y=coordinate_vector[:, 1],
                                                mode="lines",
                                                line={
                                                    "color": "gold",
                                                    "width": 4,
                                                },
                                                name=f"cell {int(cell)}"))
        first_cell = int(row[0])
        curr_frame = go.Frame(data=contour_trace,
                              name=f"cell {first_cell} + neighbours")  # TODO: check if this still works after locations switches from df to dict
        frames.append(curr_frame)

    frame_names = [frame["name"] for frame in frames]

    # create layout
    layout = layout_cell_outline_plot(layout_base, frame_names, background, d1, d2)

    # Assemble figure
    figure = go.Figure(data=frames[0]["data"],
                       layout=layout,
                       frames=frames)

    return figure
