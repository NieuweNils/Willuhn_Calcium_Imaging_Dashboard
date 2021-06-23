import base64
import io

import dash_core_components as dcc
from dash.dependencies import Output, Input, State
import pandas as pd
from scipy.io import loadmat

from app import app
from data_processing import retrieve_metadata, get_mean_locations, shortest_distances
from figures import cell_outline_chart


# TODO: check is these are neccesary
@app.callback(Output("slider", "value"),
              [Input("session_overview-animated-line-chart", "currentTime")])
def update_slider(current_time):
    return current_time


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


def get_drop_down_list(neurons_closest_together_df):
    drop_down_list = []
    for index, row in neurons_closest_together_df.iterrows():
        drop_down_list.append({'label': f'cell {int(row[0])} & cell {int(row[1])}', 'value': row[2]})
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
              ],
              [Input('locations', 'data'),
              Input('metadata', 'data')],
              )
def update_drop_down(locations, metadata):
    print("create_drop_down called ")
    locations_df = pd.read_json(locations)
    mean_locations = get_mean_locations(locations_df, metadata)
    neurons_closest_together = shortest_distances(mean_locations)
    drop_down_list_1 = get_drop_down_list(neurons_closest_together)
    drop_down_list_2 = get_drop_down_list(neurons_closest_together)
    drop_down_list_3 = get_drop_down_list(neurons_closest_together)
    return [dcc.Dropdown(id='cell-selector-drop-down-1', options=drop_down_list_1),
            dcc.Dropdown(id='cell-selector-drop-down-2', options=drop_down_list_2),
            dcc.Dropdown(id='cell-selector-drop-down-3', options=drop_down_list_3),
            ]


@app.callback(
    [
        Output('cell-shape-plot-1', 'figure'),
        Output('cell-shape-plot-2', 'figure'),
        Output('cell-shape-plot-3', 'figure'),
    ],
    [
        Input('locations-df', 'data'),
        Input('metadata', 'data'),
        Input('cell-selector-drop-down-1', 'value'),
        Input('cell-selector-drop-down-2', 'value'),
        Input('cell-selector-drop-down-3', 'value'),
    ])
def update_cell_shape_plots(locations_df, metadata, cell_1, cell_2, cell_3):
    print("update_cell_shape_plots called")
    figure1 = cell_outline_chart(locations_df, metadata, cell_1)
    figure2 = cell_outline_chart(locations_df, metadata, cell_2)
    figure3 = cell_outline_chart(locations_df, metadata, cell_3)
    return [figure1,  # output 1
            figure2,  # output 2
            figure3]  # output 3

