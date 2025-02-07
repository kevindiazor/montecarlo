from dash import Dash, dcc, html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Remove if you do not actually use it

# Load and filter the data
df = pd.read_csv("mo_salah.csv")
df = df[df['SEASON'] == '2023-24']
goals = df['G'].values
assists = df['A'].values

# Define the Monte Carlo simulation function
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

# Initialize the Dash app
app = Dash(__name__)

# Define the dashboard layout
app.layout = html.Div([
    html.H1("Mohamed Salah Season Predictor"),
    html.P("Use the slider to select the number of games to simulate:"),
    dcc.Slider(
        id='num-games-slider',
        min=10,
        max=38,
        step=1,
        value=38,
        marks={i: str(i) for i in range(10, 39)}
    ),
    html.Div(id='slider-output'),
    dcc.Graph(id='goals-distribution'),
    dcc.Graph(id='assists-distribution')
])

# Define the callback with proper Output and Input wrappers (using only one callback decorator)
from dash.dependencies import Output, Input

@app.callback(
    [Output('slider-output', 'children'),
     Output('goals-distribution', 'figure'),
     Output('assists-distribution', 'figure')],
    [Input('num-games-slider', 'value')]
)
def update_output(num_games):
    simulated_goals, simulated_assists = monte_carlo_simulation(10000, num_games)
    
    goals_mean = np.mean(simulated_goals)
    goals_5th = np.percentile(simulated_goals, 5)
    goals_95th = np.percentile(simulated_goals, 95)
    
    assists_mean = np.mean(simulated_assists)
    assists_5th = np.percentile(simulated_assists, 5)
    assists_95th = np.percentile(simulated_assists, 95)

    goals_fig = {
        'data': [{
            'x': simulated_goals,
            'type': 'histogram',
            'marker': {'color': 'blue'},
            'name': 'Goals'
        }],
        'layout': {
            'title': f'Goals Distribution (Mean: {goals_mean:.2f}, Range: {goals_5th}-{goals_95th})',
            'xaxis': {'title': 'Total Goals per Season'},
            'yaxis': {'title': 'Frequency'}
        }
    }

    assists_fig = {
        'data': [{
            'x': simulated_assists,
            'type': 'histogram',
            'marker': {'color': 'green'},
            'name': 'Assists'
        }],
        'layout': {
            'title': f'Assists Distribution (Mean: {assists_mean:.2f}, Range: {assists_5th}-{assists_95th})',
            'xaxis': {'title': 'Total Assists per Season'},
            'yaxis': {'title': 'Frequency'}
        }
    }
    
    summary_text = (f"Simulating {num_games} games per season, our model predicts Salah would score between "
                    f"{goals_5th} and {goals_95th} goals, on average around {goals_mean:.2f} goals.")
    
    return summary_text, goals_fig, assists_fig

# Run the Dash server
if __name__ == '__main__':
    app.run_server(debug=True)
