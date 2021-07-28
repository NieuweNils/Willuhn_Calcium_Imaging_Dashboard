import dash_core_components as dcc
import dash_html_components as html
# import dash_player as player
import dash_table
import numpy as np
import skimage.io as skim_io
from scipy.io import loadmat

from app import app
from figures import animated_heatmap, animated_line_chart
from formatting import colours, navbar_current_page, filter_border_style, font_family, summary_style, graph_col_style, \
    graph_row_style, upload_button_style

####################################################################################################
# 000 - DATA MAPPING
####################################################################################################

# Set datasource

# TODO: Define Data mappings


####################################################################################################
# 000 - IMPORT DATA
####################################################################################################

###########################
data_after_CNMF_E = loadmat("data/concat_may_NoRMCorre_results.mat")
data_after_CNMF_E = data_after_CNMF_E["results"]
fluorescence_traces = np.array(data_after_CNMF_E["C"][0][0])
data_line_chart = fluorescence_traces

tiff_data = skim_io.imread("data/concat_may_NoRMCorre.tif")
data_heatmap = tiff_data[0:1000]


################################################################################################################################################## SET UP END

####################################################################################################
# 000 - DEFINE REUSABLE COMPONENTS AS FUNCTIONS
####################################################################################################

#####################
# Header with logo
def get_header():
    header = html.Div([

        html.Div([], className="col-2"),  # Same as img width, allowing to have the title centrally aligned

        html.Div([
            html.H1(children="Calcium Imaging Dashboard",
                    style={"textAlign": "center"}
                    )],
            className="col-8",
            style={"padding-top": "1.5%"}
        ),

        html.Div([
            html.Img(
                src=app.get_asset_url("calcium_imaging_logo.png"),
                height="100 px",
                width="auto")
        ],
            className="col-2",
            style={
                "align-items": "center",
                "padding-top": "1%",
                "height": "auto"})

    ],
        className="row",
        style={"height": "4%",
               "background-color": colours["super-dark-green"]}
    )

    return header


#####################
# Nav bar
def get_navbar(p="session_overview"):
    style = {
        "session_overview": {},
        "double_cell_selector": {},
        "page_3": {},
    }

    if p == "session_overview":
        style["session_overview"] = navbar_current_page
    elif p == "double_cell_selector":
        style["double_cell_selector"] = navbar_current_page
    else:
        style["page_3"] = navbar_current_page

    navbar = html.Div([

        html.Div([], className="col-3"),

        html.Div([
            dcc.Link(
                html.H4(children="Session overview",
                        style=style["session_overview"]),
                href="/session_overview"
            )
        ],
            className="col-2"),

        html.Div([
            dcc.Link(
                html.H4(children="Double cell selector",
                        style=style["double_cell_selector"]),
                href="/double_cell_selector"
            )
        ],
            className="col-2"),

        html.Div([
            dcc.Link(
                html.H4(children="Optional 3rd page",
                        style=style["page_3"]),
                href="/page_3"
            )
        ],
            className="col-2"),

        html.Div([], className="col-3")

    ],
        className="row",
        style={"background-color": colours["dark-green"],
               "box-shadow": "2px 5px 5px 1px rgba(255, 101, 131, .5)"}
    )

    return navbar


#####################
# Empty row

def get_emptyrow(h="45px"):
    """This returns an empty row of a defined height"""

    emptyrow = html.Div([
        html.Div([
            html.Br()
        ], className="col-12")
    ],
        className="row",
        style={"height": h})

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
    get_navbar("session_overview"),

    #####################
    # Row 3 : Filters
    html.Div([  # External row

        html.Div([  # External 12-column

            html.Div([  # Internal row

                # Internal columns
                html.Div([
                ],
                    className="col-2"),  # Blank 2 columns

                # Filter pt 1
                html.Div([

                    html.Div([
                        html.H5(
                            children="Filters by Date:",
                            style={"text-align": "left", "color": colours["medium-blue-grey"]}
                        ),
                        # Date range picker
                        html.Div(["Select a date range: ",
                                  dcc.DatePickerRange(
                                      id="date-picker-session_overview",
                                      start_date_placeholder_text="Start date",
                                      display_format="DD-MMM-YYYY",
                                      first_day_of_week=1,
                                      end_date_placeholder_text="End date",
                                      style={"font-size": "12px", "display": "inline-block", "border-radius": "2px",
                                             "border": "1px solid #ccc", "color": "#333", "border-spacing": "0",
                                             "border-collapse": "separate"})
                                  ], style={"margin-top": "5px"}
                                 )

                    ],
                        style={"margin-top": "10px",
                               "margin-bottom": "5px",
                               "text-align": "left",
                               "paddingLeft": 5})

                ],
                    className="col-4"),  # Filter part 1

                # Filter pt 2
                html.Div([

                    html.Div([
                        html.H5(
                            children="Optional second filter:",
                            style={"text-align": "left", "color": colours["medium-blue-grey"]}
                        ),
                    ],
                        style={"margin-top": "10px",
                               "margin-bottom": "5px",
                               "text-align": "left",
                               "paddingLeft": 5})

                ],
                    className="col-4"),  # Filter part 2

                html.Div([
                ],
                    className="col-2")  # Blank 2 columns

            ],
                className="row")  # Internal row

        ],
            className="col-12",
            style=filter_border_style)  # External 12-column

    ],
        className="row sticky-top"),  # External row

    #####################
    # Row 4
    get_emptyrow(),

    #####################
    # Row 5 : Charts
    html.Div([  # External row

        html.Div([
        ],
            className="col-1"),  # Blank 1 column

        html.Div([  # External 10-column

            html.H2(children="Traces",
                    style={"color": colours["white"]}),

            html.Div([  # Internal row - RECAPS

                html.Div([], className="col-4"),  # Empty column

                html.Div([
                    dash_table.DataTable(
                        id="recap-table",
                        style_header={
                            "backgroundColor": "transparent",
                            "fontFamily": font_family,
                            "font-size": "1rem",
                            "color": colours["light-green"],
                            "border": "0px transparent",
                            "textAlign": "center"},
                        style_cell={
                            "backgroundColor": "transparent",
                            "fontFamily": font_family,
                            "font-size": "0.85rem",
                            "color": colours["white"],
                            "border": "0px transparent",
                            "textAlign": "center"},
                    )
                ],
                    className="col-4"),

                html.Div([], className="col-4")  # Empty column

            ],
                className="row",
                style=summary_style
            ),  # Internal row - RECAPS

            html.Div([  # Internal row

                # Chart Column
                html.Div([
                    dcc.Graph(
                        # TODO: make the selection criteria user defined
                        figure=animated_heatmap(data=data_heatmap, skip_rate=50),
                        id="session_overview-animated-heatmap")
                ],
                    className="col-4"),

                # Chart Column
                html.Div([
                    # player.DashPlayer(
                    #     id="session_overview-fluorescence-video-player",
                    #     url="/output.webm",
                    #     controls=True,
                    #     width="100%",
                    # )
                ],
                    className="col-4"),

                # Chart Column
                html.Div(
                    [
                        # dcc.Graph(
                        #     id="session_overview-UNUSED")
                    ],
                    className="col-4")

            ],
                className="row"),  # Internal row

            html.Div(  # Internal Row
                [
                    html.Div(  # Chart Column 1
                        [
                            dcc.Slider(
                                id="slider",
                                min=0,
                                max=10,
                                step=0.5,
                                value=0,
                                marks={
                                    0: {"label": "0s", "style": {"color": "black"}},
                                    2: {"label": "2s", "style": {"color": "black"}},
                                    4: {"label": "4s", "style": {"color": "black"}},
                                    6: {"label": "6s", "style": {"color": "black"}},
                                    8: {"label": "8s", "style": {"color": "black"}},
                                    10: {"label": "10s", "style": {"color": "black"}},
                                },
                            ),
                        ],
                        style={"width": "30%", "display": "inline-block"},
                    ),  # Chart column 1

                    html.Div(  # Chart Column 2
                        html.Br()
                    ),  # Chart column 2

                    html.Div(  # Chart Column 3
                        html.Br()
                    ),  # Chart column 3
                ]),  # Internal Row

            html.Div([  # Internal row

                # Chart Column
                html.Div([
                    dcc.Graph(
                        # TODO: make the selection criteria user defined
                        figure=animated_line_chart(data=data_line_chart[7, 0:5000]),
                        id="session_overview-animated-line-chart")
                ],
                    className="col-4"),

                # Chart Column
                html.Div([
                    # dcc.Graph(
                    #     id="session_overview-UNUSED")
                ],
                    className="col-4"),

                # Chart Column
                html.Div([
                    # dcc.Graph(
                    #     id="session_overview-UNUSED")
                ],
                    className="col-4")

            ],
                className="row")  # Internal row

        ],
            className="col-10",
            style=graph_col_style),  # External 10-column

        html.Div([
        ],
            className="col-1"),  # Blank 1 column

    ],
        className="row",
        style=graph_row_style
    ),  # External row

])

####################################################################################################
# 002 - Double Cell Selector
####################################################################################################

# TODO: Data processing is not fast enough here. Look at possible speed improvements (e.g. better caching/moving to
#  numpy). I have a feeling it's the storing of the dataframe in memory as a JSON.
double_cell_selector = html.Div([

    #####################
    # Row 1 : Header
    get_header(),

    #####################
    # Row 2 : Nav bar
    get_navbar("double_cell_selector"),

    #####################
    # Row 3 : Filters
    html.Div([  # External row

        html.Br()

    ],
        className="row sticky-top"),  # External row

    #####################
    # Row 4
    get_emptyrow(),

    #####################
    # Row 5 : Charts
    html.Div([  # External row

        html.Div([
        ],
            className="col-1"),  # Blank 1 column

        html.Div([  # External 10-column

            html.Div([
                html.Div([
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div([
                            "Drag and Drop or ",
                            html.A("Select .mat Files")
                        ]),
                        style=upload_button_style,
                        # TODO: this should probably not be possible
                        # Allow multiple files to be uploaded
                        multiple=True
                    ),
                ],
                    className="col-4",
                ),
                html.Div(id="empty-column-placeholder",
                         className="col-4"),
                html.Div(id="download-data-placeholder",
                         className="col-4")
            ],
                className="row"
            ),

            html.Div([  # start of graph field

                html.Div([
                    html.Div(id="neighbour-table",
                             className="col-4"),
                ], className="row"
                ),

                get_emptyrow(),
                html.Div([
                    html.Div(id="cell-shape-plot-1",
                             className="col-4"),
                    html.Div([dcc.Graph(id="trace-plot")],
                             className="col-4"),
                    html.Div([
                        html.Div(id="drop-down-delete-placeholder"),
                        html.Div(id="delete-button-placeholder"),
                        html.Div(id="drop-down-merge-placeholder"),
                        html.Div(id="merge-button-placeholder"),
                        html.Div(id="drop-down-traces-placeholder"),
                        html.Div(id="output-drop-down-delete"),
                    ],
                        id="actionable-buttons",
                        className="col-4"
                    ),
                ], className="row"
                ),
                # TODO: rename the intermediates to "initial", and the others to "updates"
                dcc.Store(id="locations"),
                dcc.Store(id="locations_intermediate"),
                dcc.Store(id="fluorescence_traces"),
                dcc.Store(id="fluorescence_traces_intermediate"),
                dcc.Store(id="background_fluorescence"),
                dcc.Store(id="metadata"),
                dcc.Store(id="neighbours"),
                dcc.Store(id="neighbours_intermediate"),
                dcc.Store(id="trigger-cell-shape-plot"),
            ])

        ],
            className="col-10",
            style=graph_col_style
        ),
        html.Div([],
                 className="col-1"),  # Blank 1 column

    ],
        className="row",
        style=graph_row_style
    ),  # External row
],
)

####################################################################################################

####################################################################################################
# 003 - Optional 3rd page
####################################################################################################

page_3 = html.Div([

    #####################
    # Row 1 : Header
    get_header(),

    #####################
    # Row 2 : Nav bar
    get_navbar("page_3"),

    #####################
    # Row 3 : Filters
    html.Div([  # External row

        html.Br()

    ],
        className="row sticky-top"),  # External row

    #####################
    # Row 4
    get_emptyrow(),

    #####################
    # Row 5 : Charts
    html.Div([  # External row

        html.Br()

    ])

])
