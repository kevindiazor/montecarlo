# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 22:45:13 2025

@author: Kevin
"""

import pandas as pd
import numpy as np
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import matplotlib.pyplot as plt
import io
import base64

# 1. Load and Filter the Data
# ----------------------------
# Read the CSV file of Mohamed Salah's 2023–24 season data.
df = pd.read_csv("mo_salah.csv")
# Filter the DataFrame to include only the relevant season.
df = df[df['SEASON'] == '2023-24']

# 2. Extract Empirical Game-by-Game Outcomes
# -------------------------------------------
# These arrays hold the per-game outcomes that we will re-sample from.
goals = df['G'].values       # Goals per game
assists = df['A'].values     # Assists per game

# 3. Define the Monte Carlo Simulation Function
# ------------------------------------------------
def monte_carlo_simulation(num_simulations, num_games):
    """
    For each simulated season, randomly sample 'num_games' outcomes (with replacement)
    from the historical goals and assists per game.
    """
    simulated_goals = np.array([
        np.sum(np.random.choice(goals, size=num_games, replace=True))
        for _ in range(num_simulations)
    ])
    simulated_assists = np.array([
        np.sum(np.random.choice(assists, size=num_games, replace=True))
        for _ in range(num_simulations)
    ])
    return simulated_goals, simulated_assists

# 4. Initialize the Dash App
# -----------------------------
app = dash.Dash(__name__)

# 5. Define the Dashboard Layout
# --------------------------------
app.layout = html.Div([
    html.H1("Mohamed Salah Season Predictor"),
    html.P("Use the slider to select the number of games to simulate:"),
    dcc.Slider(
        id='num-games-slider',
        min=10,
        max=38,
        step=1,
        value=38,  # Default to a full season (38 games)
        marks={i: str(i) for i in range(10, 39)}
    ),
    html.Div(id='slider-output'),
    dcc.Graph(id='goals-distribution'),
    dcc.Graph(id='assists-distribution')
])

# 6. Set Up the Callback to Update the Dashboard
# ------------------------------------------------
@app.callback(
    [Output('slider-output', 'children'),
     Output('goals-distribution', 'figure'),
     Output('assists-distribution', 'figure')],
    [Input('num-games-slider', 'value')]
)
def update_output(num_games):
    # Run the Monte Carlo simulation with 10,000 simulated seasons.
    simulated_goals, simulated_assists = monte_carlo_simulation(10000, num_games)
    
    # Calculate key summary statistics for goals.
    goals_mean = np.mean(simulated_goals)
    goals_5th = np.percentile(simulated_goals, 5)
    goals_95th = np.percentile(simulated_goals, 95)
    
    # Calculate key summary statistics for assists.
    assists_mean = np.mean(simulated_assists)
    assists_5th = np.percentile(simulated_assists, 5)
    assists_95th = np.percentile(simulated_assists, 95)
    
    # Create the histogram figure for goals using Plotly's figure format.
    goals_fig = {
        'data': [{
            'x': simulated_goals,
            'type': 'histogram',
            'marker': {'color': 'blue'},
            'name': 'Goals'
        }],
        'layout': {
            'title': f'Goals Distribution (Mean: {goals_mean:.2f}, 5th-95th: {goals_5th:.0f}-{goals_95th:.0f})',
            'xaxis': {'title': 'Total Goals per Season'},
            'yaxis': {'title': 'Frequency'}
        }
    }
    
    # Create the histogram figure for assists.
    assists_fig = {
        'data': [{
            'x': simulated_assists,
            'type': 'histogram',
            'marker': {'color': 'green'},
            'name': 'Assists'
        }],
        'layout': {
            'title': f'Assists Distribution (Mean: {assists_mean:.2f}, 5th-95th: {assists_5th:.0f}-{assists_95th:.0f})',
            'xaxis': {'title': 'Total Assists per Season'},
            'yaxis': {'title': 'Frequency'}
        }
    }
    
    # Compose a summary text based on the simulation results.
    summary_text = (f"Simulating {num_games} games per season, our model predicts that "
                    f"Salah would score between {goals_5th} and {goals_95th} goals "
                    f"in most seasons.")
    
    return summary_text, goals_fig, assists_fig

# 7. Run the Dash Server
# ------------------------
if __name__ == '__main__':
    app.run_server(debug=True)