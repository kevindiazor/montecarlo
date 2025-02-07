from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os
from functools import lru_cache
from typing import Tuple

# ---------------------------
# Liverpool FC Official Colors
# ---------------------------
class LFCColors:
    PRIMARY_RED = "#C8102E"  # Official primary red [1][4]
    SECONDARY_GREEN = "#00B2A9"  # Official accent green [1]
    GOLD_ACCENT = "#F6EB61"  # Official gold accent [1]
    WHITE = "#FFFFFF"  # Standard white [2][5]
    DARK_RED = "#5E1208"  # Dark red from logo [3]

# ---------------------------
# Configuration Constants
# ---------------------------
class Config:
    SEASON_FILTER = os.environ.get('SEASON_FILTER', '2023-24')
    NUM_SIMULATIONS = int(os.environ.get('NUM_SIMULATIONS', 10_000))
    DEFAULT_GAMES = int(os.environ.get('DEFAULT_GAMES', 38))
    CACHE_SIZE = 128
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
    marks={i: {'label': str(i), 'style': {'color': LFCColors.DARK_RED}} 
           for i in range(10, 39, 2)},
    tooltip={"placement": "bottom", "always_visible": True},
    updatemode='drag',
    className='lfc-slider'
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
            'opacity': 0.85,
            'histnorm': 'probability density',
            'hovertemplate': f'{title}: %{{x:.1f}}<extra></extra>'
        }],
        'layout': {
            'title': f'<b>{title}</b> Distribution<br>μ={mean:.1f} σ={std:.1f} (90% CI: {p5}-{p95})',
            'margin': {'t': 80, 'b': 40, 'l': 40, 'r': 40},
            'xaxis': {'title': f'Total {title}', 'gridcolor': LFCColors.WHITE},
            'yaxis': {'title': 'Probability Density', 'gridcolor': LFCColors.WHITE},
            'plot_bgcolor': LFCColors.WHITE,
            'paper_bgcolor': LFCColors.WHITE,
            'font': {'color': LFCColors.DARK_RED}
        }
    }

# ---------------------------
# App Layout with LFC Styling
# ---------------------------
app.layout = dbc.Container([
    dcc.Store(id='simulation-cache'),
    
    dbc.Row(dbc.Col(
        html.Div([
            html.H1("Mohamed Salah Performance Predictor", 
                   className="display-4",
                   style={'color': LFCColors.PRIMARY_RED}),
            html.P("Premier League 2023-24 Season Projections", 
                  className="lead",
                  style={'color': LFCColors.SECONDARY_GREEN})
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
                        'border': f'4px solid {LFCColors.PRIMARY_RED}',
                        'borderRadius': '15px',
                        'boxShadow': f'0 8px 16px {LFCColors.SECONDARY_GREEN}33',
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
                      style={'color': LFCColors.DARK_RED}),
            slider
        ], className="mb-4 p-3 border rounded",
           style={'borderColor': LFCColors.SECONDARY_GREEN})
    ])),
    
    dbc.Row(dbc.Col(
        dcc.Loading(
            html.Div(id='slider-output', 
                    className="h4 text-center mb-4",
                    style={'color': LFCColors.PRIMARY_RED}),
            type="circle",
            color=LFCColors.PRIMARY_RED
        )
    )),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='goals-distribution'), className="mb-4", xs=12, md=6),
        dbc.Col(dcc.Graph(id='assists-distribution'), className="mb-4", xs=12, md=6)
    ])
], fluid=True, style={'backgroundColor': LFCColors.WHITE})

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
        create_distribution_figure(goals_data, 'Goals', LFCColors.PRIMARY_RED),
        create_distribution_figure(assists_data, 'Assists', LFCColors.SECONDARY_GREEN)
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
