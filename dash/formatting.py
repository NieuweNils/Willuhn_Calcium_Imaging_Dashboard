import plotly.graph_objs as go

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
    'light-grey': 'rgb(208, 206, 206)',
}

graph_row_style = {
    'margin-left': '15px',
    'margin-right': '15px',
}

graph_col_style = {
    'border-radius': '10px',
    'border-style': 'solid',
    'border-width': '1px',
    'border-color': colors['superdark-green'],
    'background-color': colors['superdark-green'],
    'box-shadow': '0px 0px 17px 0px rgba(186, 218, 212, .5)',
    'padding-top': '10px',
}

filter_border_style = {
    'border-radius': '0px 0px 10px 10px',
    'border-style': 'solid',
    'border-width': '1px',
    'border-color': colors['light-green'],
    'background-color': colors['light-green'],
    'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)',
}

navbar_current_page = {
    'text-decoration': 'underline',
    'text-decoration-color': colors['pink-red'],
    'text-shadow': '0px 0px 1px rgb(251, 251, 252)',
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
    'background-color': 'rgb(251, 251, 252, 0.1)',
}

summary_text_style = {
    'text-align': 'left',
    'font-weight': '350',
    'color': colors['white'],
    'font-size': '1.5rem',
    'letter-spacing': '0.04em',
}

####################### Chart formatting

title = {
    'font': {
        'size': 16,
        'color': colors['white'], }
}

x_axis = {
    'showgrid': False,
    'linecolor': colors['light-grey'],
    'color': colors['white'],
    'tickangle': 315,
    'titlefont': {
        'size': 12,
        'color': colors['white'],
    },
    'tickfont': {
        'size': 11,
        'color': colors['white'],
    },
    'zeroline': False,
}

y_axis = {
    'showgrid': True,
    'color': colors['white'],
    'gridwidth': 0.5,
    'gridcolor': colors['white'],
    'linecolor': colors['white'],
    'titlefont': {
        'size': 12,
        'color': colors['white'],
    },
    'tickfont': {
        'size': 11,
        'color': colors['white'],
    },
    'zeroline': False,
}

font_family = 'Dosis'

legend = {
    'orientation': 'h',
    'yanchor': 'bottom',
    'y': 1.01,
    'xanchor': 'right',
    'x': 1.05,
    'font': {'size': 9,
             'color': colors['white'],
             }
}  # Legend will be on the top right, above the graph, horizontally

margins = {'l': 5, 'r': 5, 't': 45, 'b': 15}  # Set top margin to in case there is a legend

layout = go.Layout(
    font={'family': font_family},
    title=title,
    title_x=0.5,  # Align chart title to center
    paper_bgcolor=colors['superdark-green'],  # rgba(0,0,0,0)',
    plot_bgcolor=colors['white'],
    xaxis=x_axis,
    yaxis=y_axis,
    height=270,
    legend=legend,
    margin=margins
)

upload_button_style = {
    'width': '15%',
    'height': '35px',
    'lineHeight': '30px',
    'borderWidth': '1px',
    'borderStyle': 'solid',
    'borderRadius': '5px',
    'textAlign': 'center',
    'margin': '10px',
    'color': colors['white'],
}
