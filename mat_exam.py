import dash
# import dash_core_components as dcc
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State

import plotly.graph_objs as go

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(
        id='example-graph',
        figure=go.Figure(
            data=[
                go.Scatter(x=[1, 2, 3, 4, 5], y=[1, 6, 3, 6, 1], mode='lines', line=dict(width=2), name='Original Curve')
            ],
            layout=go.Layout(title='Click to Increase Line Width', clickmode='event+select')
        )
    ),
    html.Button('Click Me', id='button'),
    dcc.Store(id='memory-output', data=2)
])

@app.callback(
    Output('example-graph', 'figure'),
    [Input('button', 'n_clicks')],
    [State('example-graph', 'figure'), State('memory-output', 'data')]
)
def update_graph(n_clicks, figure, line_width):
    if n_clicks:
        figure['data'][0]['line']['width'] = line_width + 1  # increase line width
        return figure
    return figure

@app.callback(
    Output('memory-output', 'data'),
    [Input('example-graph', 'clickData')],
    [State('memory-output', 'data')]
)
def store_click_data(clickData, line_width):
    if clickData:
        return line_width + 1  # increase stored line width
    return line_width

if __name__ == '__main__':
    app.run_server(debug=True)
