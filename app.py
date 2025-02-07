from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os
from functools import lru_cache
from typing import Tuple

# ---------------------------
# Configuration Constants
# ---------------------------
class Config:
    SEASON_FILTER = os.environ.get('SEASON_FILTER', '2023-24')
    NUM_SIMULATIONS = int(os.environ.get('NUM_SIMULATIONS', 10_000))
    DEFAULT_GAMES = int(os.environ.get('DEFAULT_GAMES', 38))
    CACHE_SIZE = 128
    LIVERPOOL_RED = "#C8102E"
    LIVERPOOL_SECONDARY = "#00B2A9"
    SALAH_IMAGE = "assets/salah_image.jpg"

# ---------------------------
# Data Loading with Memoization
# ---------------------------
@lru_cache(maxsize=1)
def load_data() -> Tuple[np.ndarray, np.ndarray]:
    df = (
        pd.read_csv("mo_salah.csv", usecols=['SEASON', 'G', 'A'])
        .query("SEASON == @Config.SEASON_FILTER")
        .astype({'G': 'uint8', 'A': 'uint8'})
    )
    return df['G'].values, df['A'].values

# ---------------------------
# Vectorized Simulation with Caching
# ---------------------------
@lru_cache(maxsize=Config.CACHE_SIZE)
def monte_carlo_simulation(num_games: int) -> Tuple[np.ndarray, np.ndarray]:
    goals, assists = load_data()
    rng = np.random.default_rng()
    
    goals_sim = rng.choice(goals, (Config.NUM_SIMULATIONS, num_games)).sum(axis=1)
    assists_sim = rng.choice(assists, (Config.NUM_SIMULATIONS, num_games)).sum(axis=1)
    return goals_sim, assists_sim

# ---------------------------
# Dashboard Components
# ---------------------------
app = Dash(__name__, 
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           meta_tags=[{"viewport": "width=device-width, initial-scale=1"}],
           compress=True)

slider = dcc.Slider(
    id='num-games-slider',
    min=10,
    max=38,
    step=1,
    value=Config.DEFAULT_GAMES,
    marks={i: {'label': str(i), 'style': {'transform': 'rotate(45deg)'}} 
           for i in range(10, 39, 2)},
    tooltip={"placement": "bottom", "always_visible": True}
)

def create_distribution_figure(data: np.ndarray, title: str, color: str) -> dict:
    mean, std = data.mean(), data.std()
    p5, p95 = np.percentile(data, [5, 95])
    
    return {
        'data': [{
            'x': data,
            'type': 'histogram',
            'name': 'Count',
            'marker': {'color': color},
            'opacity': 0.7,
            'histnorm': 'probability density',
            'hovertemplate': f'{title}: %{{x:.1f}}<extra></extra>'
        }],
        'layout': {
            'title': f'<b>{title}</b> Distribution<br>μ={mean:.1f} σ={std:.1f} (90% CI: {p5}-{p95})',
            'margin': {'t': 80, 'b': 40},
            'xaxis': {'title': f'Total {title}'},
            'yaxis': {'title': 'Probability Density'},
            'plot_bgcolor': '#f8f9fa'
        }
    }

# ---------------------------
# App Layout with Image
# ---------------------------
app.layout = dbc.Container([
    dcc.Store(id='simulation-cache'),
    
    dbc.Row(dbc.Col(
        html.Div([
            html.H1("Mohamed Salah Performance Predictor", 
                   className="display-4",
                   style={'color': Config.LIVERPOOL_RED}),
            html.P("Premier League 2023-24 Season Projections", 
                  className="lead text-muted")
        ], className="text-center my-5")
    )),
    
    dbc.Row(
        dbc.Col(
            html.Div(
                html.Img(
                    src=Config.SALAH_IMAGE,
                    style={
                        'width': '100%',
                        'maxWidth': '600px',
                        'border': f'4px solid {Config.LIVERPOOL_RED}',
                        'borderRadius': '15px',
                        'boxShadow': '0 8px 16px rgba(0,0,0,0.2)',
                        'margin': '20px auto',
                        'display': 'block',
                        'position': 'relative',
                        'clipPath': 'polygon(0 0, 100% 0, 100% 90%, 97% 100%, 3% 100%, 0 90%)'
                    }
                ),
                className='position-relative text-center'
            ),
            width=12,
            className='my-4'
        )
    ),
    
    dbc.Row(dbc.Col([
        html.Div([
            html.Label("Remaining Games to Simulate:", 
                      className="h5",
                      style={'color': Config.LIVERPOOL_SECONDARY}),
            slider
        ], className="mb-4 p-3 border rounded")
    ])),
    
    dbc.Row(dbc.Col(
        dcc.Loading(
            html.Div(id='slider-output', 
                    className="h4 text-center mb-4",
                    style={'color': Config.LIVERPOOL_RED}),
            type="circle"
        )
    )),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='goals-distribution'), className="mb-4", xs=12, md=6),
        dbc.Col(dcc.Graph(id='assists-distribution'), className="mb-4", xs=12, md=6)
    ])
], fluid=True, className="py-3")

# ---------------------------
# Callback Implementation
# ---------------------------
@callback(
    [Output('slider-output', 'children'),
     Output('goals-distribution', 'figure'),
     Output('assists-distribution', 'figure')],
    Input('num-games-slider', 'value'),
    prevent_initial_call=False
)
def update_components(num_games: int) -> Tuple[str, dict, dict]:
    if not 10 <= num_games <= 38:
        raise ValueError("Number of games must be between 10 and 38")
    
    goals_data, assists_data = monte_carlo_simulation(num_games)
    
    return (
        f"Projection for {num_games} games: "
        f"{goals_data.mean():.1f}±{goals_data.std():.1f} goals | "
        f"{assists_data.mean():.1f}±{assists_data.std():.1f} assists",
        create_distribution_figure(goals_data, 'Goals', Config.LIVERPOOL_RED),
        create_distribution_figure(assists_data, 'Assists', Config.LIVERPOOL_SECONDARY)
    )

# ---------------------------
# Server Configuration
# ---------------------------
if __name__ == '__main__':
    app.run_server(
        host=os.environ.get('HOST', '0.0.0.0'),
        port=int(os.environ.get('PORT', 8050)),
        debug=os.environ.get('DEBUG', 'false').lower() == 'true',
        dev_tools_props_check=False
    )
