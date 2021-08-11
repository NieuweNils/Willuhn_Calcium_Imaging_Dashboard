from copy import copy

import dash_table
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from data_processing import get_pixel_df, get_cols_and_rows
from formatting import standard_layout, colours, font_family



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


def correlation_plot(cell_list, correlation_df, distances, min_correlation=0.2, max_distance=10, layout_base=standard_layout):
    # Discard correlation values below threshold
    highly_correlating_neurons = correlation_df[correlation_df > min_correlation]
    # Discard non-neighbours
    neighbours = distances[distances["distance"] < max_distance]
    cells_to_select_row = set(neighbours["neuron_1"])
    cells_to_select_col = set(neighbours["neuron_2"])
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


def retrieve_contours(locations, background, thr=None, threshold_method="max", maxthr=0.2, energy_threshold=0.9,
                      display_numbers=True, max_number=None,
                      cmap="gray", swap_dim=False, colors="yellow", vmin=None,
                      vmax=None, **kwargs):
    """Plots contour of spatial components against a background image
       and returns their coordinates

       From Caiman: https://github.com/flatironinstitute/CaImlocationsn
       @author: agiovann

     Parameters:
     -----------
     locations:   np.ndarray or sparse matrix
               Matrix of Spatial components (d x K)

     background:  np.ndarray (2D)
               Background image (e.g. mean, correlation)

     thr_method: [optional] string
              Method of thresholding:
                  'max' sets to zero pixels that have value less
                  than a fraction of the max value
                  'nrg' keeps the pixels that contribute up to a
                  specified fraction of the energy

     maxthr: [optional] scalar
                Threshold of max value

     nrgthr: [optional] scalar
                Threshold of energy

     thr: scalar between 0 and 1
               Energy threshold for computing contours (default 0.9)
               Kept for backwards compatibility.
               If not None then thr_method = 'nrg', and nrgthr = thr

     display_number:     Boolean
               Display number of ROIs if checked (default True)

     max_number:    int
               Display the number for only the first max_number components
               (default None, display all numbers)

     cmap:     string
               User specifies the colormap (default None, default colormap)

     Returns:
     --------
     contour_plots:     list
                A list of 3D numpy arrays containing the image in RGBA format (png compatible)
    """
    if sparse.issparse(locations):
        locations = np.array(locations.todense())
    else:
        locations = np.array(locations)

    if swap_dim:
        background = background.T
        print('Swapping dimensions')

    d1, d2 = np.shape(background)
    nr_pixels, nr_cells = np.shape(locations)
    if max_number is None:
        max_number = nr_cells

    if thr is not None:
        threshold_method = 'nrg'
        energy_threshold = thr
        warn("The way to call utilities.plot_contours has changed.")

    x, y = np.mgrid[0:d1:1, 0:d2:1]

    if vmax is None and vmin is None:
        plt.imshow(background, interpolation=None, cmap=cmap,
                   vmin=np.percentile(background[~np.isnan(background)], 1),
                   vmax=np.percentile(background[~np.isnan(background)], 99))
    else:
        plt.imshow(background, interpolation=None, cmap=cmap,
                   vmin=vmin, vmax=vmax)

    contour_plots = []
    for i in range(np.minimum(nr_cells, max_number)):
        # remove a contourplot if it was already drawn.
        for collection in plt.gca().collections:
            collection.remove()

        if threshold_method == 'nrg':
            index = np.argsort(locations[:, i], axis=None)[::-1]
            cum_energy = np.cumsum(locations[:, i].flatten()[index]**2)
            cum_energy /= cum_energy[-1]
            location_vector = np.zeros(nr_pixels)
            location_vector[index] = cum_energy
            thr = energy_threshold

        else:
            if threshold_method != 'max':
                warn("Unknown threshold method. Choosing max")
            location_vector = locations[:, i].flatten()
            location_vector /= np.max(location_vector)  # normalise location vector
            thr = maxthr

        if swap_dim:
            location_matrix = np.reshape(location_vector, np.shape(background), order='C')
        else:
            location_matrix = np.reshape(location_vector, np.shape(background), order='F')
        plt.axis('off')
        plt.contour(y, x, location_matrix, [thr], colors="orange")

        fig = plt.gcf()
        fig.tight_layout(pad=0)

        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw')
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.close()
        contour_plots.append(img_arr)

    return contour_plots
