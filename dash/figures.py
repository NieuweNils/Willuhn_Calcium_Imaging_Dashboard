from copy import copy

import dash_table
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from data_processing import get_pixel_df, get_cols_and_rows, correlating_neurons
from formatting import standard_layout, colours, font_family


# HELPER FUNCTIONS

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
            "prefix": "Cell: ",
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
                 'label': f'Cell {name}',
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


# CELL LOCATIONS


def transform_data_cell_outline_plot(locations_df, metadata):
    d1 = metadata['d1']  # height of recording in #pixels
    d2 = metadata['d2']  # width of recording in #pixels
    pixel_df = get_pixel_df(locations_df)
    cols, rows = get_cols_and_rows(pixel_df, metadata)
    neuron_positions = np.stack((rows, cols))
    number_of_cells = neuron_positions.shape[1]

    return neuron_positions, number_of_cells, d1, d2


def find_pixels(neuron_positions, cell_number):
    neuron_position = neuron_positions[:, cell_number, :]
    neuron_position = neuron_position[~np.isnan(neuron_position)]
    amount_of_pixels = int(neuron_position.shape[0] / 2)
    neuron_position = np.reshape(neuron_position, (2, amount_of_pixels))

    return neuron_position, amount_of_pixels


def make_outline_image(amount_of_pixels, neuron_position, d1, d2, starting_image=None, value_if_present=2.0):
    if starting_image is None:
        transparent_background = np.full([d1, d2], np.nan)
        image_neuron = transparent_background
    else:
        image_neuron = np.array(starting_image)  # TODO: see if this step can be skipped by an overlay of heatmaps

    for j in range(amount_of_pixels):
        row = int(neuron_position[0, j])
        col = int(neuron_position[1, j])
        image_neuron[row, col] = value_if_present
    return image_neuron


def gray_heatmap(image, layout_base=standard_layout):
    return go.Figure(
            data=go.Heatmap(
                z=image,
                colorscale='gray'),
            layout=layout_base,
            )


def frames_cell_outline_plot(number_of_cells, neuron_positions, d1, d2, background):
    frames = []
    frame_names = []
    for cell_number in range(number_of_cells):
        neuron_position, amount_of_pixels = find_pixels(neuron_positions, cell_number)
        image_neuron = make_outline_image(amount_of_pixels,
                                          neuron_position,
                                          d1,
                                          d2,
                                          starting_image=background)
        frame_name = str(cell_number)
        curr_frame = go.Frame(data=[go.Heatmap(z=image_neuron,
                                               colorscale=[
                                                   [0, 'rgb(0, 0, 0)'],  # black background
                                                   [0.1, 'rgb(50, 50, 50)'],
                                                   [0.2, 'rgb(100, 100, 100)'],
                                                   [0.3, 'rgb(150, 150, 150)'],
                                                   [0.4, 'rgb(200, 200, 200)'],
                                                   [0.5, 'rgb(250, 250, 250)'],  # white background
                                                   [1.0, 'rgb(255,255,0)'],  # yellow (for highlighted cell)
                                               ],
                                               )],
                              name=frame_name)  # VERY IMPORTANT: FRAME NAMES==SLIDER STEP NAMES==DROP DOWN NAMES
        frames.append(curr_frame)
        frame_names.append(frame_name)

    return frames, frame_names


def layout_cell_outline_plot(layout, frame_names, d1, d2):
    slider_dict = slider_base()
    slider_dict["steps"] = slider_steps(frame_names)
    layout["sliders"] = [slider_dict]

    drop_down_settings = drop_down(frame_names)
    layout["updatemenus"] = [drop_down_settings]

    layout["xaxis"]["range"] = [0, d2]
    layout["yaxis"]["range"] = [0, d1]

    return layout


def cell_outlines(locations_df, metadata, background=None, layout_base=standard_layout):
    """
    :param locations_df: a pandas dataframe with locations of each neuron, as stored in the the matlab variable "A"
    in the output of the CNMF_E algorithm
    :param metadata: a dictionary with all "options" variables, retrieved from the the matlab variable "options" in the
     output of the CNMF_E algorithm
    :param background:
    :param layout_base: a custom layout (    either a dict or a plotly layout object). default is a standard layout,
    imported from the "formatting" library
    :return: a plotly.Figure object containing plotly.graph_obj.Heatmap objects displaying the location of the neurons
    """
    layout_base = copy(layout_base)  # Copy by value instead of reference
    # create frames
    neuron_positions, number_of_cells, d1, d2 = transform_data_cell_outline_plot(locations_df, metadata)
    frames, frame_names = frames_cell_outline_plot(number_of_cells, neuron_positions, d1, d2, background)
    first_frame = frames[0]['data']
    # create layout
    layout = layout_cell_outline_plot(layout_base, frame_names, d1, d2)
    # Assemble figure
    figure = go.Figure(data=first_frame,
                       layout=layout,
                       frames=frames)

    return figure


def update_cell_outlines(figure_dict, cells_to_be_deleted):
    # make a list of the cells that should remain after the deletion
    cells_to_keep = [int(frame['name']) for frame in figure_dict["frames"]]
    for cell in cells_to_be_deleted:
        cells_to_keep.remove(cell)
    # keep the setting for the cells_to_keep (==discard those for cells_to_be_deleted)
    print("deleting frames from figure")
    for fig in [figure_dict]:
        fig["frames"] = [frame for frame in fig["frames"] if int(frame["name"]) in cells_to_keep]
        fig["data"][0] = fig["frames"][0]["data"][0]
        steps = fig["layout"]["sliders"][0]["steps"]
        buttons = fig["layout"]["updatemenus"][0]["buttons"]  # the drop down menu
        fig["layout"]["sliders"][0]["steps"] = [step for step in steps if int(step["label"]) in cells_to_keep]
        fig["layout"]["updatemenus"][0]["buttons"] = [btn for btn in buttons if int(btn["label"][5:]) in cells_to_keep]

        print("making an updated figure")
    return go.Figure(figure_dict)


def frames_double_cells_outline_plot(neuron_positions, neighbours_df, d1, d2):
    frames = []
    frame_names = []
    cells_with_neighbours = neighbours_df['neuron']

    for index, cell_number in enumerate(cells_with_neighbours):
        # make image of the cell itself
        neuron_position, amount_of_pixels = find_pixels(neuron_positions, cell_number)
        image_neuron = make_outline_image(amount_of_pixels,
                                          neuron_position,
                                          d1,
                                          d2)
        # add neighbours in gray
        neighbours = neighbours_df[neighbours_df['neuron'] == cell_number].values
        neighbours = neighbours[~pd.isnull(neighbours)]
        neighbours = neighbours[1:, ]  # drop first column, that's the neuron itself
        for neighbour in neighbours:
            position_neighbour, amount_of_pixels_neighbour = find_pixels(neuron_positions, int(neighbour))
            image_including_neighbours = make_outline_image(amount_of_pixels_neighbour,
                                                            position_neighbour,
                                                            d1,
                                                            d2,
                                                            starting_image=image_neuron,
                                                            value_if_present=0.7)
        frame_name = str(cell_number)
        curr_frame = go.Frame(data=[go.Heatmap(z=image_including_neighbours,
                                               colorscale=[
                                                   [0, 'rgb(0, 0, 0)'],  # black background
                                                   [0.1, 'rgb(50, 50, 50)'],
                                                   [0.2, 'rgb(100, 100, 100)'],
                                                   [0.3, 'rgb(150, 150, 150)'],
                                                   [0.4, 'rgb(200, 200, 200)'],
                                                   [0.5, 'rgb(250, 250, 250)'],  # white background
                                                   [1.0, 'rgb(255,255,0)'],  # yellow (for highlighted cell)
                                               ],)],
                              name=frame_name)  # VERY IMPORTANT: FRAME NAMES==SLIDER STEP NAMES==DROP DOWN NAMES
        frames.append(curr_frame)
        frame_names.append(frame_name)

    return frames, frame_names


def cell_outlines_double_cells(locations_df, neighbours_df, metadata, layout_base=standard_layout):
    """
    :param locations_df: a pandas dataframe with locations of each neuron, as stored in the the matlab variable "A" in the output of the CNMF_E algorithm
    :param neighbours_df: a pandas dataframe with for each cell that has close neighbouring cells, which cells those are
    :param metadata: a dictionary with all "options" variables, retrieved from  the the matlab variable "options" in the output of the CNMF_E algorithm
    :param layout_base: a custom layout (either a dict or a plotly layout object). default is a standard layout, imported from the "formatting" library
    :return: a plotly.Figure object containing plotly.graph_obj.Heatmap objects displaying the location of the neurons
    """
    layout_base = copy(layout_base)  # Copy by value instead of reference
    # create frames
    neuron_positions, _, d1, d2 = transform_data_cell_outline_plot(locations_df, metadata)
    frames, frame_names = frames_double_cells_outline_plot(neuron_positions, neighbours_df, d1, d2)
    first_frame = frames[0]['data']
    # create layout
    layout = layout_cell_outline_plot(layout_base, frame_names, d1, d2)
    # Assemble figure
    figure = go.Figure(data=first_frame, layout=layout, frames=frames)
    return figure


def correlation_plot(fluorescence_traces):
    correlation_matrix = np.corrcoef(fluorescence_traces)
    correlation_matrix = np.absolute(correlation_matrix)
    correlation_df = pd.DataFrame(correlation_matrix)
    # Discard the lower left triangle as all correlation values will end up as doubles (includes the diagonal of 1.0's)
    correlation_df = correlation_df.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
    highly_correlating_neurons = correlation_df[correlation_df > 0.6]
    highly_correlating_neurons.dropna(how='all')
    #
    # return dash_table.DataTable(id='neurons-close-together-datatable',
    #                      columns=[table_columns],
    #                      data=table_data,
    #                      fixed_rows={'headers': True},
    #                      style_header={
    #                          'backgroundColor': 'transparent',
    #                          'fontFamily': font_family,
    #                          'font-size': '1rem',
    #                          'color': colours['light-green'],
    #                          'border': '0px transparent',
    #                          'textAlign': 'center',
    #                      },
    #                      style_table={
    #                          'height': '300px',
    #                          'width': '600px',
    #                          'marginLeft': '5%',
    #                          'marginRight': 'auto',
    #                          'overflowY': 'auto',
    #                      },
    #                      style_cell={
    #                          'backgroundColor': colours['dark-green'],
    #                          'color': colours['white'],
    #                          'border': '0px transparent',
    #                          'textAlign': 'center',
    #                      }
    figure = go.Figure(data=[go.Heatmap(z=highly_correlating_neurons)])

    return figure


def line_chart(cells_to_display, traces, layout_base=standard_layout):
    """
    :param traces: a 2d numpy array of dimensions [channel, timestamp] that stores fluorescence traces for each neuron
    :param layout_base: settings to use in plotly.graph_objs.figure['layout']
    :return: a line chart object (of type plotly.graph_objs.figure where keys ['traces'] stores plotly.graph_objs.Scatter
     objects)
    """
    figure = go.Figure()
    stop = len(traces[0])
    x_axis = list(range(0, stop))
    number_of_cells = len(traces)
    cells_to_display = sorted(cells_to_display)
    # TODO: Track the cell number of each of the traces
    # TODO: (Otherwise I will be in trouble once cells are deleted & merged,
    # TODO: because the indexes will not be the same after that)
    traces_to_display = [traces[index] for index in list(range(number_of_cells)) if index in cells_to_display]
    for index, trace in enumerate(traces_to_display):
        figure.add_trace(go.Scatter(x=x_axis, y=trace[0:stop], name="cell " + str(cells_to_display[index]), mode="lines"))

    layout = copy(layout_base)  # Copy by value, NOT by reference
    layout["xaxis"]["title"] = "time (ms)"
    layout["xaxis"]["range"] = [0, x_axis[-1]]
    layout["yaxis"]["title"] = "intensity of signal"
    layout["yaxis"]["range"] = [0, np.max([scatter["y"] for scatter in figure["data"]])]
    layout["autosize"] = False

    figure.layout = layout

    return figure

