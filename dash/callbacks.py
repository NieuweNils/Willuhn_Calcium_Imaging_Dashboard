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
from figures import cell_outline_chart
from formatting import colours

# TODO: check if this is necessary
@app.callback(Output("slider", "value"),
              [Input("session_overview-animated-line-chart", "currentTime")])
def update_slider(current_time):
    return current_time


# TODO: check if this is necessary
@app.callback(Output("video-player", "playing"),
              [Input("play-button", "n_clicks")],
              [State("session_overview-animated-line-chart", "playing")],)
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
        return [locations_df.to_json(),  # output 1
                metadata]      # output 2
    return [None, None]


@app.callback([
                Output('drop-down-selector-1', 'children'),
                Output('drop-down-selector-2', 'children'),
                Output('drop-down-selector-3', 'children'),
                Output('neurons-close-together', 'children'),
                ],
              [Input('locations', 'data'),
              Input('metadata', 'data')],
              prevent_initial_call=True,
              )
def initialise_new_data_screen(locations, metadata):
    print("update_drop_down called ")
    start_time = time.time()
    locations_df = pd.read_json(locations)
    mean_locations = get_mean_locations(locations_df, metadata)
    neurons_closest_together = shortest_distances(mean_locations)
    neighbour_df = a_neurons_neighbours(neurons_closest_together)
    duration = time.time()-start_time
    print(f"the data part above took {duration}s")
    drop_down_list = get_drop_down_list(neighbour_df)
    drop_down_list_2 = get_drop_down_list (neighbour_df)
    drop_down_list_3 = get_drop_down_list(neighbour_df)
    table_columns = [{"name": i, "id": i} for i in neurons_closest_together.columns]
    table_data = neurons_closest_together.to_dict('records')
    return [dcc.Dropdown(id='cell-selector-drop-down-1', options=drop_down_list),
            dcc.Dropdown(id='cell-selector-drop-down-2', options=drop_down_list_2),
            dcc.Dropdown(id='cell-selector-drop-down-3', options=drop_down_list_3),
            dash_table.DataTable(id='neurons-close-together-table',
                                 columns=table_columns,
                                 data=table_data,
                                 fixed_rows={'headers': True},
                                 style_table={'height': '300px',
                                              'width': '600px',
                                              'marginLeft': 'auto',
                                              'marginRight': 'auto',
                                              'overflowY': 'auto'},
                                 style_cell={'backgroundColor': colours['dark-green'],
                                             'color': colours['white']}
                                 )
            ]


# @app.callback(
#     [
#         Output('cell-shape-plot-1', 'figure'),
#         Output('cell-shape-plot-2', 'figure'),
#         Output('cell-shape-plot-3', 'figure'),
#     ],
#     [
#         Input('locations-df', 'data'),
#         Input('metadata', 'data'),
#         Input('cell-selector-drop-down-1', 'value'),  # TODO: change to State?
#         Input('cell-selector-drop-down-2', 'value'),  # TODO: change to State?
#         Input('cell-selector-drop-down-3', 'value'),  # TODO: change to State?
#     ])
# def update_cell_shape_plots(locations, metadata, cell_1, cell_2, cell_3):
#     print("update_cell_shape_plots called")
#     locations_df = pd.read_json(locations)
#     figure1 = cell_outline_chart(locations_df, metadata, cell_1)
#     figure2 = cell_outline_chart(locations_df, metadata, cell_2)
#     figure3 = cell_outline_chart(locations_df, metadata, cell_3)
#     return [figure1,  # output 1
#             figure2,  # output 2
#             figure3]  # output 3


# @app.callback(
#     Output('check-locations', 'children'),
#     [
#         Input('locations', 'data'),
#         Input('metadata', 'data'),
#     ],
#
# )
# def check_locations(locations, metadata):
#     print('Listener to Inputs "locations-data" & "metadata-data" is picking up the update')
#     if locations:
#         return f'I loaded the locations& metadata. Here is the metadata: {metadata}'
#     else:
#         return "I wasn't able to load any data"
