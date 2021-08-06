import base64
import io
import os
import time
import json

import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
from dash import callback_context
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from scipy.io import loadmat, savemat


from app import app
from data_processing import retrieve_metadata, get_mean_locations, distances, correlating_neurons, a_neurons_neighbours, \
    delete_locations, delete_traces, delete_neighbours, delete_neurons_distances, merge_locations, merge_traces
from figures import cell_outlines, line_chart, correlation_plot
from formatting import colours, font_family, upload_button_style


# TODO: check if this is necessary
@app.callback(Output("slider", "value"),
              [Input("session_overview-animated-line-chart", "currentTime")])
def update_slider(current_time):
    return current_time


# TODO: check if this is necessary
@app.callback(Output("video-player", "playing"),
              [Input("play-button", "n_clicks")],
              [State("session_overview-animated-line-chart", "playing")], )
def play_video(n_clicks, playing):
    if n_clicks:
        return not playing

    return playing


def parse_data(contents, filename):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if "mat" in filename:
            # Assume that the user uploaded a .mat file
            data_after_cnmf_e = loadmat(
                io.BytesIO(decoded)
            )["results"]
            locations = data_after_cnmf_e["A"][0][0].todense()
            fluorescence_traces = np.array(data_after_cnmf_e["C_raw"][0][0])
            background_fluorescence = np.array(data_after_cnmf_e["Cn"][0][0])
            options = retrieve_metadata(data_after_cnmf_e)
    except Exception as e:
        print(e)
        return None
    return locations, fluorescence_traces, background_fluorescence, options


@app.callback([Output("locations_intermediate", "data"),
               Output("fluorescence_traces_intermediate", "data"),
               Output("background_fluorescence", "data"),
               Output("metadata", "data"),
               Output("list_of_cells_intermediate", "data"),
               Output("distance_intermediate", "data"),
               Output("correlations_intermediate", "data"),
               Output("neighbours_intermediate", "data"),
               Output("startup_trigger", "data"),
               ],
              [Input("upload-data", "contents")],
              [State("upload-data", "filename")],
              prevent_initial_call=True,
              )
def upload_data(list_of_contents, list_of_names):
    print("upload_data called ")
    if list_of_contents is not None:
        print("parsing data")
        start_time = time.time()

        locations, fluorescence_traces, background_fluorescence, metadata = parse_data(list_of_contents[0],
                                                                                       list_of_names[0])
        duration = time.time() - start_time
        print(f"parsing data took {duration}s")

        print("transforming the data")
        start_time = time.time()

        locations_df = pd.DataFrame(locations)
        mean_locations = get_mean_locations(locations_df, metadata)
        distance_df = distances(mean_locations)
        correlation_df = correlating_neurons(fluorescence_traces)
        neighbour_df = a_neurons_neighbours(distance_df, correlation_df)

        list_of_cells = list(range(len(fluorescence_traces)))
        trace_dict = {}
        for cell in list(range(len(fluorescence_traces))):
            trace_dict[cell] = fluorescence_traces[cell]

        duration = time.time() - start_time
        print(f"transforming took {duration}s")

        # NB!!!!! Do not change traces to a list that does not track the cell number associated with each of the traces
        # (Or the line chart stops working once cells are deleted & merged (indices will change upon del/merge)
        return [locations_df.to_json(),
                trace_dict,
                background_fluorescence,
                metadata,
                list_of_cells,
                distance_df.to_json(),
                correlation_df.to_json(),
                neighbour_df.to_json(),
                True,
                ]
    return [None, None, None, None, None, None, None, None]


@app.callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    [State("locations", "data"),
     State("fluorescence_traces", "data"),
     State("background_fluorescence", "data"),
     State("metadata", "data"),
     State("neighbours", "data"),
     ],
    prevent_initial_call=True,
)
def download_data(n_clicks, locations, traces, background, metadata, neighbours):
    if n_clicks is None:
        raise PreventUpdate
    loc_array = pd.read_json(locations).to_numpy(dtype="float", na_value=np.nan)
    nb_array = pd.read_json(neighbours).to_numpy(dtype="float", na_value=np.nan)
    processed_data = {
        "locations": loc_array,  # this should be a list for speed optimization (no dataframe.to_json)
        "traces": traces,
        "background": background,
        "metadata": metadata,
        "neighbours": nb_array,
    }
    savemat("processed_data.mat", mdict=processed_data)
    return dcc.send_file("processed_data.mat")



@app.callback(Output("distance-and-correlation-placeholder", "children"),
              Input("startup_trigger", "data"),
              )
def create_customization_inputs(trigger):
    if trigger:
        return html.Div([dcc.Input(id="distance_criteria",
                                   placeholder='max distance (pixels)',
                                   type='number',
                                   value=''),
                         dcc.Input(id="correlation_criteria",
                                   placeholder='min correlation coeff',
                                   type='number',
                                   value='',
                                   min=0, max = 1, step = 0.1,
                                   ),
                         html.Button("update neighbours",
                                     id="neighbour-criteria-button",
                                     style=upload_button_style
                                     ),
                         ]
                        )


# TODO: if I store the index of the cell in the neighbour_df, it"ll be easier to merge & delete entries in the dashboard
def get_drop_down_list(list_of_cells):
    drop_down_list = []
    for cell in list_of_cells:
        drop_down_list.append({"label": f"cell {int(cell)}", "value": int(cell)})
    sorted_drop_down = sorted(drop_down_list, key=lambda list_entry: list_entry["value"])
    return sorted_drop_down


# TODO: change this to make use of the rows of neighbouring cells
@app.callback([
    Output("drop-down-delete-placeholder", "children"),
    Output("drop-down-merge-placeholder", "children"),
    Output("drop-down-traces-placeholder", "children"),
    Output("trigger-cell-shape-plot", "data"),  # TODO: check if this is the right place to call this
],
    [Input("list_of_cells_intermediate", "data"),
     Input("list_of_cells", "data")],
    prevent_initial_call=True,
)
def update_drop_downs(uploaded_data, cached_data):
    print("update_drop_downs called ")

    if cached_data is None:
        if uploaded_data is not None:
            cells = uploaded_data
    else:
        cells = cached_data

    drop_down_list = get_drop_down_list(cells)

    return [dcc.Dropdown(id="drop-down-delete", options=drop_down_list, multi=True,
                         placeholder="Select cells to delete"),
            dcc.Dropdown(id="drop-down-merge", options=drop_down_list, multi=True,
                         placeholder="Select cells to merge"),
            dcc.Dropdown(id="drop-down-traces", options=drop_down_list, multi=True,
                         placeholder="Select cells to show their traces"),
            {"value": True},  # hack to trigger cell shape plot the first time
            ]


@app.callback(
    [Output("delete-button-placeholder", "children"),
     Output("merge-button-placeholder", "children")],
    Input("neighbours_intermediate", "modified_timestamp"),
    prevent_initial_call=True,
)
def create_delete_and_merge_buttons(timestamp):
    return [html.Button("Delete selected cells", id="delete-button", style=upload_button_style),
            html.Button("Merge selected cells", id="merge-button", style=upload_button_style)]


# TODO: split this up into several callbacks
@app.callback(
    [
        Output("locations", "data"),
        Output("fluorescence_traces", "data"),
        Output("neighbours", "data"),
        Output("list_of_cells", "data"),
        Output("distance", "data"),
        Output("correlations", "data"),
    ],
    [
        Input("delete-button", "n_clicks"),
        Input("merge-button", "n_clicks"),
        Input("neighbour-criteria-button", "n_clicks"),
    ],
    [
        State("locations_intermediate", "data"),
        State("fluorescence_traces_intermediate", "data"),
        State("neighbours_intermediate", "data"),
        State("list_of_cells_intermediate", "data"),
        State("distance_intermediate", "data"),
        State("correlations_intermediate", "data"),

        State("drop-down-delete", "value"),
        State("drop-down-merge", "value"),
        State("distance_criteria", "value"),
        State("correlation_criteria", "value"),

        State("locations", "data"),
        State("fluorescence_traces", "data"),
        State("neighbours", "data"),
        State("list_of_cells", "data"),
        State("distance", "data"),
        State("correlations", "data"),

        State("metadata", "data"),
    ],
    prevent_initial_call=True
)
def update_data_stores(n_clicks_del, n_clicks_merge, n_clicks_criteria,
                       uploaded_loc, uploaded_traces, uploaded_nb, uploaded_cell_list, uploaded_distance, uploaded_correlations,
                       cells_to_be_deleted, cells_to_be_merged, distance, correlation,
                       cached_loc, cached_traces, cached_nb, cached_cell_list, cached_distance, cached_correlations,
                       metadata):
    print("update_data_stores called")
    # if there is no data in the Stores, use the uploaded data
    if cached_loc is None or cached_traces is None or cached_nb is None or cached_cell_list is None or cached_distance is None or cached_correlations is None:
        (cached_loc, cached_traces, cached_nb, cached_cell_list, cached_distance, cached_correlations) = uploaded_loc, uploaded_traces, uploaded_nb, uploaded_cell_list, uploaded_distance, uploaded_correlations

    # there is already data in cache, and no one clicked a button:
    ctx = callback_context
    if ctx.triggered[0]["prop_id"].split(".")[0] == "delete-button":
        if n_clicks_del is None:
            print("no button clicks, raising PreventUpdate.")
            raise PreventUpdate
        # delete the cells
        if cells_to_be_deleted:
            locations_df = pd.read_json(cached_loc)
            neighbours_df = pd.read_json(cached_nb)
            distance_df = pd.read_json(cached_distance)
            updated_locations = delete_locations(df=locations_df, delete_list=cells_to_be_deleted).to_json()
            updated_traces = delete_traces(trace_dict=cached_traces, delete_list=cells_to_be_deleted)
            updated_cell_list = list(updated_traces.keys())
            updated_neighbours = delete_neighbours(df=neighbours_df, delete_list=cells_to_be_deleted).to_json()
            updated_neuron_distance = delete_neurons_distances(df=distance_df,
                                                                delete_list=cells_to_be_deleted).to_json()
            return [updated_locations, updated_traces, updated_neighbours, updated_cell_list, updated_neuron_distance, cached_correlations]
        else:
            print("no cells to be deleted, raising PreventUpdate")
            raise PreventUpdate
    if ctx.triggered[0]["prop_id"].split(".")[0] == "merge-button":
        if n_clicks_merge is None:
            print("no button clicks, raising PreventUpdate.")
            raise PreventUpdate
        if cells_to_be_merged:
            locations_df = pd.read_json(cached_loc)
            neighbours_df = pd.read_json(cached_nb)
            distance_df = pd.read_json(cached_distance)
            updated_locations = merge_locations(locations=locations_df, merge_list=cells_to_be_merged).to_json()
            updated_traces = merge_traces(traces=cached_traces, merge_list=cells_to_be_merged)
            updated_cell_list = list(updated_traces.keys())
            updated_neighbours = delete_neighbours(df=neighbours_df, delete_list=cells_to_be_merged[1:]).to_json()
            updated_neuron_distance = delete_neurons_distances(df=distance_df,
                                                                delete_list=cells_to_be_merged[1:]).to_json()
            return [updated_locations, updated_traces, updated_neighbours, updated_cell_list, updated_neuron_distance, cached_correlations]
        else:
            print("no cells to be merged, raising PreventUpdate")
            raise PreventUpdate
    if ctx.triggered[0]["prop_id"].split(".")[0] == "neighbour-criteria-button":
        if n_clicks_criteria is None:
            print("no button clicks, raising PreventUpdate.")
            raise PreventUpdate
        locations_df = pd.read_json(cached_loc)
        mean_locations = get_mean_locations(locations_df, metadata)
        distance_df = distances(mean_locations)
        correlation_df = pd.read_json(cached_correlations)
        neighbour_df = a_neurons_neighbours(distance_df, correlation_df,
                                            max_distance=distance if distance else 10,
                                            min_correlation=float(correlation) if correlation else 0.1)

        updated_distance = distance_df.to_json()
        updated_neighbours = neighbour_df.to_json()

        return [cached_loc, cached_traces, updated_neighbours, cached_cell_list, updated_distance, cached_correlations]


@app.callback(
    Output("download-data-placeholder", "children"),
    Input("neighbours", "modified_timestamp"),
    prevent_initial_call=True,
)
def update_download_button(timestamp_neighbours):
    print("update_download_button called")
    return html.Div(
        [html.Button("Download data",
                     id="download-button",
                     style=upload_button_style),
         dcc.Download(id="download-data"),
         ],
    )


@app.callback(Output("neighbour-table", "children"),
              [Input("neighbours_intermediate", "data"),
               Input("neighbours", "modified_timestamp"),
               ],
              State("neighbours", "data"),
              prevent_initial_call=True,
              )
def update_neighbour_table(nb_upload, timestamp, nb_update):
    ctx = callback_context
    if ctx.triggered[0]["prop_id"].split(".")[0] == "neighbours_intermediate":
        print("creating neighbour table for the first time")
        neighbour_df = pd.read_json(nb_upload)
    elif ctx.triggered[0]["prop_id"].split(".")[0] == "neighbours":
        print("pushing update to neighbour table")
        neighbour_df = pd.read_json(nb_update)
    else:
        print("update_neighbour_table was triggered by an unknown trigger, raising PreventUpdate")
        raise PreventUpdate

    table_columns = [{"name": i, "id": i} for i in neighbour_df.columns]
    table_data = neighbour_df.to_dict("records")

    return dash_table.DataTable(id="neighbour-datatable",
                                columns=table_columns,
                                data=table_data,
                                fixed_rows={"headers": True},
                                style_header={
                                    "backgroundColor": "transparent",
                                    "fontFamily": font_family,
                                    "font-size": "1rem",
                                    "color": colours["light-green"],
                                    "border": "0px transparent",
                                    "textAlign": "center",
                                },
                                style_table={
                                    "height": "100%",
                                    "width": "100%",
                                    "marginLeft": "0%",
                                    "marginRight": "auto",
                                    "overflowY": "auto",
                                },
                                style_cell={
                                    "backgroundColor": colours["dark-green"],
                                    "color": colours["white"],
                                    "border": "0px transparent",
                                    "textAlign": "center",
                                }
                                )


@app.callback(
    Output("correlation-plot", "children"),
    [Input("distance_intermediate", "data"),
     Input("distance", "data"),
     Input("correlations_intermediate", "data"),
     Input("correlations", "data"),

     Input("neighbour-criteria-button", "n_clicks")
     ],
    [State("list_of_cells", "data"),
     State("distance_criteria", "value"),
     State("correlation_criteria", "value"),
     ],
    prevent_inital_call=True,
)
def update_correlation_plot(dist_uploaded, dist_cached, cor_uploaded, cor_cached,
                            n_clicks,
                            cell_list, distance, correlation):
    correlations = cor_uploaded if (cor_cached is None) else cor_cached
    distance_table = dist_uploaded if (dist_cached is None) else dist_cached

    if correlations is not None and distance_table is not None:
        distance_df = pd.read_json(distance_table)
        correlation_df = pd.read_json(correlations)
        figure = correlation_plot(cell_list, correlation_df, distance_df,
                                  min_correlation=float(correlation) if correlation else 0.1,
                                  max_distance=distance if distance else 10)
        return dcc.Graph(figure=figure,
                         style={'width': '100%',
                                'height': '100%',
                                'margin': 'auto'})
    else:
        raise PreventUpdate


# TODO: find out why this is not always triggered (and I think never by locations_intermediate)
@app.callback(
    [Output("cell-shape-plot-1", "children")],
    [Input("trigger-cell-shape-plot", "data"),
     Input("delete-button", "n_clicks")],
    [State("locations_intermediate", "data"),
     State("background_fluorescence", "data"),
     State("metadata", "data"),
     State("drop-down-delete", "value"),
     State("cell-shape-plot-1", "children")],
)
def update_cell_shape_plots(trigger, n_clicks,
                            locations, background_fluorescence, metadata,
                            cells_to_be_deleted,
                            cell_shape_plot,
                            ):
    beginning = time.time()
    ctx = callback_context
    if locations is not None and cell_shape_plot is None:
        print("creating figures for the first time")
        start_time = time.time()
        locations_df = pd.read_json(locations)
        duration = time.time() - start_time
        print(f"the data took {duration}s to load into a dataframe")

        start_time = time.time()
        cell_outline_figure = cell_outlines(locations_df, metadata, background=background_fluorescence)
        duration = time.time() - start_time
        print(f"that figure took {duration}s to make")
        ending = time.time()
        duration = ending - beginning
        print(f"this function call took {duration}s")
        return [dcc.Graph(id="cell_outline_graph",
                          figure=cell_outline_figure),
                ]

    if ctx.triggered[0]['prop_id'].split('.')[0] == "delete-button":
        raise PreventUpdate
        # if n_clicks is None:
        #     print("no delete button clicks, raising PreventUpdate.")
        #     raise PreventUpdate
        # if cells_to_be_deleted is None:
        #     print("no cells selected for deletion, raising PreventUpdate")
        #     raise PreventUpdate
        # figure_settings = cell_shape_plot["props"]["figure"]
        # updated_figure = update_cell_outlines(figure_settings, cells_to_be_deleted)
        # print("Pushing an update to the figures")
        # # TODO: the updating of the graph doesn't seem to work... FIX THIS!
        # return [dcc.Graph(id="cell_outline_graph",
        #                   figure=updated_figure)
        #         ]
    else:
        print("update_cell_shape_plots was triggered by an unknown, unexpected trigger. raising PreventUpdate")
        raise PreventUpdate


@app.callback(
    Output("trace-plot", "figure"),
    Input("drop-down-traces", "value"),
    State("fluorescence_traces_intermediate", "data"),
    State("fluorescence_traces", "data"),
    prevent_initial_call=True
)
def update_trace_plot(cells_to_display, traces_uploaded, traces_cached):
    print("update_trace_plot called")
    if not cells_to_display or traces_uploaded is None:
        raise PreventUpdate
    if not traces_cached:
        traces = traces_uploaded
    else:
        traces = traces_cached
    trace_plot = line_chart(cells_to_display, traces)

    return trace_plot
