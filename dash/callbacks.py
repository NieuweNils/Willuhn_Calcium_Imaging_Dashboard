import base64
import io
import time

import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
from dash.dependencies import Output, Input, State
from scipy.io import loadmat

from app import app
from data_processing import retrieve_metadata, get_mean_locations, shortest_distances, a_neurons_neighbours
from figures import cell_outlines, cell_outlines_double_cells, gray_heatmap
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


def get_drop_down_list(neighbours_df):
    drop_down_list = []
    for index, row in neighbours_df.iterrows():
        drop_down_list.append({'label': f'cell {int(row[0])}', 'value': int(row[0])})
    return drop_down_list


@app.callback(
    Output("download-data", "children"),
    [Input('locations', 'data'),
     Input('neighbours', 'data')],
    prevent_initial_call=True,
)
def update_download_button(locations, neighbours):
    print("update_download_button called")
    return html.Div(
        [html.Button("Download data",
                     id="download-button",
                     style=upload_button_style),
         dcc.Download(id='download-data'),
         ],
        className='col-4'
    )


@app.callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_data(n_clicks):
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 1, 5, 6], "c": ["x", "x", "y", "y"]})
    return dcc.send_data_frame(df.to_csv, "my_df.csv")


def parse_data(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'mat' in filename:
            # Assume that the user uploaded a .mat file
            data_after_cnmf_e = loadmat(
                io.BytesIO(decoded)
            )['results']
            locations = data_after_cnmf_e['A'][0][0].todense()
            fluorescence_traces = np.array(data_after_cnmf_e['C'][0][0])
            background_fluorescence = np.array(data_after_cnmf_e['Cn'][0][0])
            options = retrieve_metadata(data_after_cnmf_e)
    except Exception as e:
        print(e)
        return None
    return locations, fluorescence_traces, background_fluorescence, options


@app.callback([Output('locations', 'data'),
               Output('fluorescence_traces', 'data'),
               Output('background_fluorescence', 'data'),
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
        locations, fluorescence_traces, background_fluorescence, metadata = parse_data(list_of_contents[0],
                                                                                       list_of_names[0])
        print("transforming the data")
        locations_df = pd.DataFrame(locations)
        mean_locations = get_mean_locations(locations_df, metadata)
        neurons_closest_together = shortest_distances(mean_locations)
        neighbour_df = a_neurons_neighbours(neurons_closest_together)
        return [locations,
                fluorescence_traces,
                background_fluorescence,
                metadata,
                neurons_closest_together.to_json(),
                neighbour_df.to_json(),
                ]
    return [None, None, None, None, None, None]


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


@app.callback(
    [
        Output('cell-shape-plot-1', 'figure'),
        Output('cell-shape-plot-2', 'figure'),
        Output('background-plot', 'figure'),
    ],
    [
        Input('locations', 'data'),
        Input('neighbours', 'data'),
        Input('background_fluorescence', 'data'),
        Input('metadata', 'data'),
        # Input('cell-selector-drop-down-1', 'value'),  # TODO: change to State?
    ],
    prevent_initial_call=True
)
def update_cell_shape_plots(locations, neighbours, background_fluorescence, metadata):
    print("update_cell_shape_plots called")

    start_time = time.time()
    locations_df = pd.DataFrame(locations)
    neighbours_df = pd.read_json(neighbours)
    duration = time.time() - start_time
    print(f"the data took {duration}s to load into a dataframe")

    start_time = time.time()
    cell_outline_figure = cell_outlines(locations_df, metadata, background=background_fluorescence)
    duration = time.time() - start_time
    print(f"that figure took {duration}s to make")

    start_time = time.time()
    double_cell_figure = cell_outlines_double_cells(locations_df, neighbours_df, metadata)
    duration = time.time() - start_time
    print(f"that figure took {duration}s to make")

    start_time = time.time()
    background_plot = gray_heatmap(background_fluorescence)
    duration = time.time() - start_time
    print(f"that figure took {duration}s to make")

    return [cell_outline_figure,
            double_cell_figure,
            background_plot]
