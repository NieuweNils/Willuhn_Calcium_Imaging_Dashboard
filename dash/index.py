import dash
import dash_core_components as dcc
import dash_html_components as html

from app import app
from layouts import session_overview, double_cell_selector, page_3

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/session_overview':
        return session_overview
    elif pathname == '/double_cell_selector':
        return double_cell_selector
    elif pathname == '/page_3':
        return page_3
    else:
        return session_overview  # This is the "home page"


if __name__ == '__main__':
    app.run_server(debug=False)
