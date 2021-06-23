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


@app.callback(
    Output("video-player", "playing"),
    [Input("play-button", "n_clicks")],
    [State("session_overview-animated-line-chart", "playing")],
)
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


@app.callback([Output('drop-down-selector', 'children'),
               Output('cell-shape-plot', 'figure')],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')], prevent_initial_call=True
              )
def update_drop_down(list_of_contents, list_of_names):
    if list_of_contents is not None:
        locations_df, metadata = parse_data(list_of_contents[0], list_of_names[0])
        mean_locations = get_mean_locations(locations_df, metadata)
        neurons_closest_together = shortest_distances(mean_locations)
        drop_down_list = get_drop_down_list(neurons_closest_together)
        figure = cell_outline_chart(locations_df, metadata, 1)
        return [dcc.Dropdown(id='cell-selector-drop-down',  # output 1
                             options=drop_down_list),
                figure]  # output 2
    return None


@app.callback(
    Output('drop-down-selection-value', 'children'),
    [Input('cell-selector-drop-down', 'value')])
def update_drop_down_value(value):
    return f'These cells have {value} pixels distance between their centres'
