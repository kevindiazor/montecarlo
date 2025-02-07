from dash import Dash, dcc, html
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os

# Load the dataset and filter to the 2023-24 season
df = pd.read_csv("mo_salah.csv")
df = df[df['SEASON'] == '2023-24']
goals = df['G'].values
assists = df['A'].values

# Monte Carlo simulation function (bootstraps season outcomes)
def monte_carlo_simulation(num_simulations, num_games):
    simulated_goals = np.array([
        np.sum(np.random.choice(goals, size=num_games, replace=True))
        for _ in range(num_simulations)
    ])
    simulated_assists = np.array([
        np.sum(np.random.choice(assists, size=num_games, replace=True))
        for _ in range(num_simulations)
    ])
    return simulated_goals, simulated_assists

# Link a Bootstrap stylesheet and set a viewport meta tag for mobile responsiveness
external_stylesheets = [dbc.themes.BOOTSTRAP]
meta_tags = [{
    "name": "viewport",
    "content": "width=device-width, initial-scale=1, shrink-to-fit=no"
}]

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=external_stylesheets, meta_tags=meta_tags)

# Define a responsive layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Mohamed Salah Season Predictor"), width=12)
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(html.P("Use the slider to select the number of games to simulate:"), width=12)
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(
            dcc.Slider(
                id='num-games-slider',
                min=10,
                max=38,
                step=1,
                value=38,
                marks={i: str(i) for i in range(10, 39)},
                tooltip={"always_visible": True}
            ),
            width=12
        )
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(html.Div(id='slider-output'), width=12)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='goals-distribution'), xs=12, sm=12, md=6),
        dbc.Col(dcc.Graph(id='assists-distribution'), xs=12, sm=12, md=6)
    ], className="mb-3")
], fluid=True)

# Update the charts and text based on slider input
@app.callback(
    [Output('slider-output', 'children'),
     Output('goals-distribution', 'figure'),
     Output('assists-distribution', 'figure')],
    [Input('num-games-slider', 'value')]
)
def update_output(num_games):
    simulated_goals, simulated_assists = monte_carlo_simulation(10000, num_games)
    
    # Calculate key statistics
    goals_mean = np.mean(simulated_goals)
    goals_5th = np.percentile(simulated_goals, 5)
    goals_95th = np.percentile(simulated_goals, 95)
    
    assists_mean = np.mean(simulated_assists)
    assists_5th = np.percentile(simulated_assists, 5)
    assists_95th = np.percentile(simulated_assists, 95)
    
    # Create a histogram figure for goals
    goals_fig = {
        'data': [{
            'x': simulated_goals,
            'type': 'histogram',
            'marker': {'color': 'blue'},
            'name': 'Goals'
        }],
        'layout': {
            'title': {
                'text': f'Goals Distribution (Mean: {goals_mean:.2f}, Range: {goals_5th}-{goals_95th})',
                'font': {'size': 20},
                'x': 0.5, 'xanchor': 'center'
            },
            'xaxis': {'title': 'Total Goals per Season'},
            'yaxis': {'title': 'Frequency'},
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)'
        }
    }
    
    # Create a similar histogram figure for assists
    assists_fig = {
        'data': [{
            'x': simulated_assists,
            'type': 'histogram',
            'marker': {'color': 'green'},
            'name': 'Assists'
        }],
        'layout': {
            'title': {
                'text': f'Assists Distribution (Mean: {assists_mean:.2f}, Range: {assists_5th}-{assists_95th})',
                'font': {'size': 20},
                'x': 0.5, 'xanchor': 'center'
            },
            'xaxis': {'title': 'Total Assists per Season'},
            'yaxis': {'title': 'Frequency'},
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)'
        }
    }
    
    summary_text = (f"Simulating {num_games} games per season, our model predicts Salah would score between "
                    f"{goals_5th} and {goals_95th} goals, on average around {goals_mean:.2f} goals.")
    
    return summary_text, goals_fig, assists_fig

# Run the app – note that we use host=0.0.0.0 so that the app is accessible externally
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=True)
