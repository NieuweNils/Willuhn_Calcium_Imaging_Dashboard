import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go
from dash.development.base_component import Component

from app import app

####################################################################################################
# 000 - FORMATTING INFO
####################################################################################################

####################### CSS formatting
colors = {
    'dark-blue-grey': 'rgb(62, 64, 76)',
    'medium-blue-grey': 'rgb(77, 79, 91)',
    'superdark-green': 'rgb(41, 56, 55)',
    'dark-green': 'rgb(57, 81, 85)',
    'medium-green': 'rgb(93, 113, 120)',
    'light-green': 'rgb(186, 218, 212)',
    'pink-red': 'rgb(255, 101, 131)',
    'dark-pink-red': 'rgb(247, 80, 99)',
    'white': 'rgb(251, 251, 252)',
    'light-grey': 'rgb(208, 206, 206)'
}

graph_row_style = {
    'margin-left': '15px',
    'margin-right': '15px'
}

graph_col_style = {
    'border-radius': '10px',
    'border-style': 'solid',
    'border-width': '1px',
    'border-color': colors['superdark-green'],
    'background-color': colors['superdark-green'],
    'box-shadow': '0px 0px 17px 0px rgba(186, 218, 212, .5)',
    'padding-top': '10px'
}

filter_border_style = {
    'border-radius': '0px 0px 10px 10px',
    'border-style': 'solid',
    'border-width': '1px',
    'border-color': colors['light-green'],
    'background-color': colors['light-green'],
    'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'
}

navbar_current_page = {
    'text-decoration': 'underline',
    'text-decoration-color': colors['pink-red'],
    'text-shadow': '0px 0px 1px rgb(251, 251, 252)'
}

summary_style = {
    'border-radius': '10px',
    'border-style': 'solid',
    'border-width': '1px',
    'border-color': 'rgb(251, 251, 252, 0.1)',
    'margin-left': '15px',
    'margin-right': '15px',
    'margin-top': '15px',
    'margin-bottom': '15px',
    'padding-top': '5px',
    'padding-bottom': '5px',
    'background-color': 'rgb(251, 251, 252, 0.1)'
}

summary_text_style = {
    'text-align': 'left',
    'font-weight': '350',
    'color': colors['white'],
    'font-size': '1.5rem',
    'letter-spacing': '0.04em'
}

####################### Corporate chart formatting

title = {
    'font': {
        'size': 16,
        'color': colors['white']}
}

x_axis = {
    'showgrid': False,
    'linecolor': colors['light-grey'],
    'color': colors['light-grey'],
    'tickangle': 315,
    'titlefont': {
        'size': 12,
        'color': colors['light-grey']},
    'tickfont': {
        'size': 11,
        'color': colors['light-grey']},
    'zeroline': False
}

y_axis = {
    'showgrid': True,
    'color': colors['light-grey'],
    'gridwidth': 0.5,
    'gridcolor': colors['dark-green'],
    'linecolor': colors['light-grey'],
    'titlefont': {
        'size': 12,
        'color': colors['light-grey']},
    'tickfont': {
        'size': 11,
        'color': colors['light-grey']},
    'zeroline': False
}

font_family = 'Dosis'

legend = {
    'orientation': 'h',
    'yanchor': 'bottom',
    'y': 1.01,
    'xanchor': 'right',
    'x': 1.05,
    'font': {'size': 9, 'color': colors['light-grey']}
}  # Legend will be on the top right, above the graph, horizontally

margins = {'l': 5, 'r': 5, 't': 45, 'b': 15}  # Set top margin to in case there is a legend

layout = go.Layout(
    font={'family': font_family},
    title=title,
    title_x=0.5,  # Align chart title to center
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=x_axis,
    yaxis=y_axis,
    height=270,
    legend=legend,
    margin=margins
)

####################################################################################################
# 000 - DATA MAPPING
####################################################################################################

# Set datasource
session_overview_datasource = 'data/datasource.xlsx'


# TODO: Define Data mappings


####################################################################################################
# 000 - IMPORT DATA
####################################################################################################

###########################
# TODO: Import data

################################################################################################################################################## SET UP END

####################################################################################################
# 000 - DEFINE REUSABLE COMPONENTS AS FUNCTIONS
####################################################################################################

#####################
# Header with logo
def get_header():
    header = html.Div([

        html.Div([], className='col-2'),  # Same as img width, allowing to have the title centrally aligned

        html.Div([
            html.H1(children='Calcium Imaging Dashboard',
                    style={'textAlign': 'center'}
                    )],
            className='col-8',
            style={'padding-top': '1.5%'}
        ),

        html.Div([
            html.Img(
                src=app.get_asset_url('calcium_imaging_logo.png'),
                height='100 px',
                width='auto')
        ],
            className='col-2',
            style={
                'align-items': 'center',
                'padding-top': '1%',
                'height': 'auto'})

    ],
        className='row',
        style={'height': '4%',
               'background-color': colors['superdark-green']}
    )

    return header


#####################
# Nav bar
def get_navbar(p='session_overview'):
    style = {
        'session_overview': {},
        'double_cell_selector': {},
        'page_3': {},
    }

    if p == 'session_overview':
        style['session_overview'] = navbar_current_page
    elif p == 'double_cell_selector':
        style['double_cell_selector'] = navbar_current_page
    else:
        style['page_3'] = navbar_current_page

    navbar = html.Div([

            html.Div([], className='col-3'),

            html.Div([
                dcc.Link(
                    html.H4(children='Session overview',
                            style=style['session_overview']),
                    href='/session_overview'
                )
            ],
                className='col-2'),

            html.Div([
                dcc.Link(
                    html.H4(children='Double cell selector',
                            style=style['double_cell_selector']),
                    href='/double_cell_selector'
                )
            ],
                className='col-2'),

            html.Div([
                dcc.Link(
                    html.H4(children='Optional 3rd page',
                            style=style['page_3']),
                    href='/page_3'
                )
            ],
                className='col-2'),

            html.Div([], className='col-3')

        ],
            className='row',
            style={'background-color': colors['dark-green'],
                   'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
        )

    return navbar


#####################
# Empty row

def get_emptyrow(h='45px'):
    """This returns an empty row of a defined height"""

    emptyrow = html.Div([
        html.Div([
            html.Br()
        ], className='col-12')
    ],
        className='row',
        style={'height': h})

    return emptyrow


####################################################################################################
# 001 - PLOT 1Z
####################################################################################################

session_overview = html.Div([

    #####################
    # Row 1 : Header
    get_header(),

    #####################
    # Row 2 : Nav bar
    get_navbar('session_overview'),

    #####################
    # Row 3 : Filters
    html.Div([  # External row

        html.Div([  # External 12-column

            html.Div([  # Internal row

                # Internal columns
                html.Div([
                ],
                    className='col-2'),  # Blank 2 columns

                # Filter pt 1
                html.Div([

                    html.Div([
                        html.H5(
                            children='Filters by Date:',
                            style={'text-align': 'left', 'color': colors['medium-blue-grey']}
                        ),
                        # Date range picker
                        html.Div(['Select a date range: ',
                                  dcc.DatePickerRange(
                                      id='date-picker-session_overview',
                                      start_date_placeholder_text='Start date',
                                      display_format='DD-MMM-YYYY',
                                      first_day_of_week=1,
                                      end_date_placeholder_text='End date',
                                      style={'font-size': '12px', 'display': 'inline-block', 'border-radius': '2px',
                                             'border': '1px solid #ccc', 'color': '#333', 'border-spacing': '0',
                                             'border-collapse': 'separate'})
                                  ], style={'margin-top': '5px'}
                                 )

                    ],
                        style={'margin-top': '10px',
                               'margin-bottom': '5px',
                               'text-align': 'left',
                               'paddingLeft': 5})

                ],
                    className='col-4'),  # Filter part 1

                # Filter pt 2
                html.Div([

                    html.Div([
                        html.H5(
                            children='Optional second filter:',
                            style={'text-align': 'left', 'color': colors['medium-blue-grey']}
                        ),
                    ],
                        style={'margin-top': '10px',
                               'margin-bottom': '5px',
                               'text-align': 'left',
                               'paddingLeft': 5})

                ],
                    className='col-4'),  # Filter part 2

                html.Div([
                ],
                    className='col-2')  # Blank 2 columns

            ],
                className='row')  # Internal row

        ],
            className='col-12',
            style=filter_border_style)  # External 12-column

    ],
        className='row sticky-top'),  # External row

    #####################
    # Row 4
    get_emptyrow(),

    #####################
    # Row 5 : Charts
    html.Div([  # External row

        html.Div([
        ],
            className='col-1'),  # Blank 1 column

        html.Div([  # External 10-column

            html.H2(children="Traces",
                    style={'color': colors['white']}),

            html.Div([  # Internal row - RECAPS

                html.Div([], className='col-4'),  # Empty column

                html.Div([
                    dash_table.DataTable(
                        id='recap-table',
                        style_header={
                            'backgroundColor': 'transparent',
                            'fontFamily': font_family,
                            'font-size': '1rem',
                            'color': colors['light-green'],
                            'border': '0px transparent',
                            'textAlign': 'center'},
                        style_cell={
                            'backgroundColor': 'transparent',
                            'fontFamily': font_family,
                            'font-size': '0.85rem',
                            'color': colors['white'],
                            'border': '0px transparent',
                            'textAlign': 'center'},
                    )
                ],
                    className='col-4'),

                html.Div([], className='col-4')  # Empty column

            ],
                className='row',
                style=summary_style
            ),  # Internal row - RECAPS

            html.Div([  # Internal row

                # Chart Column
                html.Div([
                    dcc.Graph(
                        id='session_overview-count-day')
                ],
                    className='col-4'),

                # Chart Column
                html.Div([
                    dcc.Graph(
                        id='session_overview-count-month')
                ],
                    className='col-4'),

                # Chart Column
                html.Div([
                    dcc.Graph(
                        id='session_overview-weekly-heatmap')
                ],
                    className='col-4')

            ],
                className='row'),  # Internal row

            html.Div([  # Internal row

                # Chart Column
                html.Div([
                    dcc.Graph(
                        id='session_overview-count-country')
                ],
                    className='col-4'),

                # Chart Column
                html.Div([
                    dcc.Graph(
                        id='session_overview-bubble-county')
                ],
                    className='col-4'),

                # Chart Column
                html.Div([
                    dcc.Graph(
                        id='session_overview-count-city')
                ],
                    className='col-4')

            ],
                className='row')  # Internal row

        ],
            className='col-10',
            style=graph_col_style),  # External 10-column

        html.Div([
        ],
            className='col-1'),  # Blank 1 column

    ],
        className='row',
        style=graph_row_style
    ),  # External row

])

####################################################################################################
# 002 - Page 2
####################################################################################################

double_cell_selector = html.Div([

    #####################
    # Row 1 : Header
    get_header(),

    #####################
    # Row 2 : Nav bar
    get_navbar('double_cell_selector'),

    #####################
    # Row 3 : Filters
    html.Div([  # External row

        html.Br()

    ],
        className='row sticky-top'),  # External row

    #####################
    # Row 4
    get_emptyrow(),

    #####################
    # Row 5 : Charts
    html.Div([  # External row

        html.Br()

    ])

])

####################################################################################################
# 003 - Optional 3rd page
####################################################################################################

page_3 = html.Div([

    #####################
    # Row 1 : Header
    get_header(),

    #####################
    # Row 2 : Nav bar
    get_navbar('page_3'),

    #####################
    # Row 3 : Filters
    html.Div([  # External row

        html.Br()

    ],
        className='row sticky-top'),  # External row

    #####################
    # Row 4
    get_emptyrow(),

    #####################
    # Row 5 : Charts
    html.Div([  # External row

        html.Br()

    ])

])
