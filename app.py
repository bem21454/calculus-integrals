from dash import Dash, html, dcc, Input, Output, callback
import plotly.graph_objs as go
import pandas as pd

app = Dash()

names = [
    'Carbon dioxide',
    'Methane',
    'Nitrous oxide',
    'Fluorinated gases',
    'Land use and forestry carbon stock change',
    'Net total',
    'Gross total'
]

app.layout = html.Div(children=[

    html.H1([
        'GreenHouse Gas Emission Analysis'
    ],
        style={'color': 'white', 'margin-left': '25px'}
    ),

    # Dropdown Selection Box
    html.Div([
        dcc.Dropdown(
            names,
            names[0],
            id='dataset',
        ),
    ]),

    # Graph
    dcc.Graph(
        id='graphic'
    ),
    html.Div([

        # Integral Input and Output
        dcc.RangeSlider(
            1950,
            2050,
            step=1,
            value=[1950, 2050],
            marks={str(_): str(_) for _ in [1950 + 10 * foo for foo in range(0, 11)]},
            tooltip={'placement': 'bottom', 'always_visible': True},
            id='int_range',
            allowCross=False,
            className='slider'
        ),
        html.Div(id='int_output', className='int_out')],
        className='integral'
    )
])


@callback(
    Output('graphic', 'figure'),
    Input('dataset', 'value'),
)
def update_graph(dataset):
    df = pd.read_csv(f'data/{dataset}.csv')
    df_filter = df.loc[df['year'].apply(lambda x: x == int(x))]
    df_actual = pd.read_csv(f'data/{dataset}_raw.csv')

    fig = go.Figure([

        # Actual data
        go.Scatter(
            name='Actual Emissions',
            x=df_actual['year'],
            y=df_actual['emission'],
            mode='markers',
            marker=dict(color='#1a5fb4', size=8)
        ),

        # Data Visuals
        go.Scatter(
            name='Mean Prediction',
            x=df['year'],
            y=df['emission'],
            mode='lines',
            hoverinfo='skip',
            line=dict(color='#153c96', width=3)
        ),
        go.Scatter(
            name='95% Confidence Interval',
            x=list(df['year']) + list(df['year'])[::-1],
            y=list(df['emission'] + 1.96 * df['std_dev']) + list(df['emission'] - 1.96 * df['std_dev'])[::-1],
            fill='toself',
            fillcolor='rgba(150,150,150,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip'
        ),

        # Hover Data
        go.Scatter(
            name='Predicted Emissions',
            x=df_filter['year'],
            y=df_filter['emission'],
            mode='none',
            showlegend=False,
        ),
        go.Scatter(
            name='Upper Bound',
            x=df_filter['year'],
            y=df_filter['emission'] + 1.96 * df_filter['std_dev'],
            mode='none',
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=df_filter['year'],
            y=df_filter['emission'] - 1.96 * df_filter['std_dev'],
            mode='none',
            showlegend=False
        ),
    ],
        layout={
            'plot_bgcolor': '#f8f8f8',
            'paper_bgcolor': '#323232',
            'font': {
                'color': 'white'
            }
        }
    )
    fig.update_layout(
        yaxis=dict(title=dict(text='Emissions (MMT)')),
        xaxis=dict(title=dict(text='Year')),
        xaxis_range=[1950, 2050],
        hovermode="x"
    )
    return fig


@callback(
    Output('int_output', 'children'),
    Input('int_range', 'value'),
    Input('dataset', 'value'),
)
def compute_integral(int_range, dataset):

    # Left Riemann Sum
    df = pd.read_csv(f'data/{dataset}.csv')
    valid_years = df.loc[df['year'].apply(lambda x: int_range[0] <= x < int_range[1])]
    return f'{round(sum(valid_years['emission']) * ((int_range[1] - int_range[0]) / len(valid_years['emission'])), 3)} MMT'


if __name__ == '__main__':
    app.run(debug=True)
