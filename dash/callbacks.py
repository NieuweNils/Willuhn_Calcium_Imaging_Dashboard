import base64
import io
import time

import dash_core_components as dcc
import dash_table
from dash.dependencies import Output, Input, State
import pandas as pd
from scipy.io import loadmat

from app import app
from data_processing import retrieve_metadata, get_mean_locations, shortest_distances, a_neurons_neighbours
from figures import cell_outlines
from formatting import colours, font_family


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
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'mat' in filename:
            # Assume that the user uploaded a .mat file
            data_after_cnmf_e = loadmat(
                io.BytesIO(decoded)
            )['results']
            locations_df = pd.DataFrame(data_after_cnmf_e['A'][0][0].todense())
            options = retrieve_metadata(data_after_cnmf_e)
    except Exception as e:
        print(e)
        return None
    return locations_df, options


def get_drop_down_list(neighbours_df):
    drop_down_list = []
    for index, row in neighbours_df.iterrows():
        drop_down_list.append({'label': f'cell {int(row[0])}', 'value': int(row[0])})
    return drop_down_list


@app.callback([Output('locations', 'data'),
               Output('metadata', 'data'),
               Output('neurons_closest_together', 'data'),
               Output('neighbours', 'data'),
               ],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')],
              prevent_initial_call=True,
              )
def load_data(list_of_contents, list_of_names):
    print("load_data called ")
    if list_of_contents is not None:
        print("parsing data")
        locations_df, metadata = parse_data(list_of_contents[0], list_of_names[0])
        print("transforming the data")
        mean_locations = get_mean_locations(locations_df, metadata)
        neurons_closest_together = shortest_distances(mean_locations)
        neighbour_df = a_neurons_neighbours(neurons_closest_together)
        return [locations_df.to_json(),
                metadata,
                neurons_closest_together.to_json(),
                neighbour_df.to_json(),
                ]
    return [None, None, None, None]


@app.callback([Output('neurons-close-together-table', 'children')],
              [Input('neurons_closest_together', 'data')],
              prevent_initial_call=True,
              )
def create_distance_table(neurons_closest_together):
    neurons_closest_together = pd.read_json(neurons_closest_together)
    table_columns = [{"name": i, "id": i} for i in neurons_closest_together.columns]
    table_data = neurons_closest_together.to_dict('records')

    return [dash_table.DataTable(id='neurons-close-together-datatable',
                                 columns=table_columns,
                                 data=table_data,
                                 fixed_rows={'headers': True},
                                 style_header={
                                     'backgroundColor': 'transparent',
                                     'fontFamily': font_family,
                                     'font-size': '1rem',
                                     'color': colours['light-green'],
                                     'border': '0px transparent',
                                     'textAlign': 'center',
                                 },
                                 style_table={
                                     'height': '300px',
                                     'width': '600px',
                                     'marginLeft': '5%',
                                     'marginRight': 'auto',
                                     'overflowY': 'auto',
                                 },
                                 style_cell={
                                     'backgroundColor': colours['dark-green'],
                                     'color': colours['white'],
                                     'border': '0px transparent',
                                     'textAlign': 'center',
                                 }
                                 )
            ]


@app.callback([Output('neighbour-table', 'children')],
              [Input('neighbours', 'data')],
              prevent_initial_call=True,
              )
def create_neighbour_table(neighbours):
    neighbour_df = pd.read_json(neighbours)
    table_columns = [{"name": i, "id": i} for i in neighbour_df.columns]
    table_data = neighbour_df.to_dict('records')

    return [dash_table.DataTable(id='neighbour-datatable',
                                 columns=table_columns,
                                 data=table_data,
                                 fixed_rows={'headers': True},
                                 style_header={
                                     'backgroundColor': 'transparent',
                                     'fontFamily': font_family,
                                     'font-size': '1rem',
                                     'color': colours['light-green'],
                                     'border': '0px transparent',
                                     'textAlign': 'center',
                                 },
                                 style_table={
                                     'height': '300px',
                                     'width': '600px',
                                     'marginLeft': '5%',
                                     'marginRight': 'auto',
                                     'overflowY': 'auto',
                                 },
                                 style_cell={
                                     'backgroundColor': colours['dark-green'],
                                     'color': colours['white'],
                                     'border': '0px transparent',
                                     'textAlign': 'center',
                                 }
                                 )
            ]


@app.callback([
    Output('drop-down-selector-1', 'children'),
    Output('drop-down-selector-2', 'children'),
    Output('drop-down-selector-3', 'children'),
],
    [Input('neighbours', 'data')],
    prevent_initial_call=True,
)
def create_drop_downs(neighbours):
    print("create_drop_downs called ")
    start_time = time.time()
    neighbour_df = pd.read_json(neighbours)

    duration = time.time() - start_time
    print(f"the data part above took {duration}s")
    drop_down_list = get_drop_down_list(neighbour_df)
    drop_down_list_2 = get_drop_down_list(neighbour_df)
    drop_down_list_3 = get_drop_down_list(neighbour_df)
    return [dcc.Dropdown(id='cell-selector-drop-down-1', options=drop_down_list),
            dcc.Dropdown(id='cell-selector-drop-down-2', options=drop_down_list_2),
            dcc.Dropdown(id='cell-selector-drop-down-3', options=drop_down_list_3),
            ]


@app.callback(
    [
        Output('cell-shape-plot-1', 'figure'),
    ],
    [
        Input('locations', 'data'),
        Input('metadata', 'data'),
        # Input('cell-selector-drop-down-1', 'value'),  # TODO: change to State?
    ],
    prevent_initial_call=True
)
def update_cell_shape_plots(locations, metadata):
    print("update_cell_shape_plots called")

    start_time = time.time()
    locations_df = pd.read_json(locations)
    duration = time.time() - start_time
    print(f"the data took {duration}s to load into a dataframe")

    start_time = time.time()
    cell_outline_figure = cell_outlines(locations_df, metadata)
    duration = time.time() - start_time
    print(f"that figure took {duration}s to make")
    return [cell_outline_figure]

#
# # Make the slider and dropdown sync for the double-cell-selector
# @app.callback(Output("dd-slider-sync", "data"),
#               [Input("cell-selector-drop-down-1", "value")],
#               prevent_initial_call=True)
# def sync_dd_value(value):
#     return value
#
#
# @app.callback(Output("cell-shape-plot-1", "data"),
#               [Input("dd-slider-sync", "data")],
#               prevent_initial_call=True)
# def sync_slider_value(value):
#     return value
#
#
# # @app.callback(Output("print-for-no-reason", "children"),
# #               [Input("cell-shape-plot-1", "data")],
# #               prevent_initial_call=True)
# # def print_plot_value(value):
# #     print("plot_value: " + str(value))
# #     return value
#
#
# @app.callback([Output("input", "value"), Output("slider", "value")], [Input("dd-slider-sync", "data")],
#               [State("input", "value"), State("slider", "value")])
# def update_components(current_value, input_prev, slider_prev):
#     # Update only inputs that are out of sync (this step "breaks" the circular dependency).
#     input_value = current_value if current_value != input_prev else dash.no_update
#     slider_value = current_value if current_value != slider_prev else dash.no_update
#     return [input_value, slider_value]
