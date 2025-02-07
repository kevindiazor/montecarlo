from dash import Dash, dcc, html, Input, Output, callback, no_update
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
    PRIMARY_RED = "#C8102E"
    SECONDARY_GREEN = "#00B2A9"
    WHITE = "#FFFFFF"
    DARK_RED = "#5E1208"

# ---------------------------
# Configuration Constants
# ---------------------------
class Config:
    SEASON_FILTER = os.environ.get('SEASON_FILTER', '2023-24')
    NUM_SIMULATIONS = int(os.environ.get('NUM_SIMULATIONS', 1000))  # Reduced for testing
    DEFAULT_GAMES = int(os.environ.get('DEFAULT_GAMES', 38))
    CACHE_SIZE = 128
    SALAH_IMAGE = "assets/salah_image.jpg"

# ---------------------------
# Data Loading with Validation
# ---------------------------
@lru_cache(maxsize=1)
def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load data with absolute path and validation"""
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, 'mo_salah.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = (
        pd.read_csv(csv_path, usecols=['SEASON', 'G', 'A'])
        .query("SEASON == @Config.SEASON_FILTER")
        .astype({'G': 'uint8', 'A': 'uint8'})
    )
    
    if df.empty:
        raise ValueError("No data found for the specified season filter")
        
    return df['G'].values, df['A'].values

# ---------------------------
# Simulation with Error Handling
# ---------------------------
@lru_cache(maxsize=Config.CACHE_SIZE)
def monte_carlo_simulation(num_games: int) -> Tuple[np.ndarray, np.ndarray]:
    """Run simulation with input validation"""
    if num_games < 1 or num_games > 38:
        raise ValueError("Number of games must be between 1 and 38")
    
    goals, assists = load_data()
    rng = np.random.default_rng()
    
    try:
        goals_sim = rng.choice(goals, (Config.NUM_SIMULATIONS, num_games)).sum(axis=1)
        assists_sim = rng.choice(assists, (Config.NUM_SIMULATIONS, num_games)).sum(axis=1)
    except Exception as e:
        raise RuntimeError(f"Simulation failed: {str(e)}")
        
    return goals_sim, assists_sim

# ---------------------------
# Dashboard Setup
# ---------------------------
app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           meta_tags=[{"viewport": "width=device-width, initial-scale=1"}])

slider = dcc.Slider(
    id='num-games-slider',
    min=10,
    max=38,
    step=1,
    value=Config.DEFAULT_GAMES,
    marks={i: {'label': str(i), 'style': {'color': LFCColors.DARK_RED}} 
           for i in range(10, 39, 2)},
    tooltip={"placement": "bottom", "always_visible": True}
)

def create_distribution_figure(data: np.ndarray, title: str, color: str) -> dict:
    """Create figure with empty data handling"""
    if data.size == 0:
        return {
            'data': [],
            'layout': {
                'title': f'No {title} Data Available',
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'annotations': [{
                    'text': 'Data Not Loaded',
                    'showarrow': False,
                    'font': {'size': 18, 'color': LFCColors.PRIMARY_RED}
                }]
            }
        }
    
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
# App Layout with Debug Elements
# ---------------------------
app.layout = dbc.Container([
    dcc.Store(id='debug-store'),
    html.Div(id='hidden-error', style={'display': 'none'}),
    
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
                        'display': 'block'
                    }
                ),
                className='text-center'
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
# Enhanced Callback with Error Handling
# ---------------------------
@callback(
    [Output('slider-output', 'children'),
     Output('goals-distribution', 'figure'),
     Output('assists-distribution', 'figure'),
     Output('hidden-error', 'children')],
    Input('num-games-slider', 'value')
)
def update_components(num_games: int):
    """Main callback with comprehensive error handling"""
    try:
        if num_games is None:
            return no_update, no_update, no_update, "No slider value received"
            
        goals_data, assists_data = monte_carlo_simulation(num_games)
        
        return (
            f"Projection for {num_games} games: "
            f"{goals_data.mean():.1f}±{goals_data.std():.1f} goals | "
            f"{assists_data.mean():.1f}±{assists_data.std():.1f} assists",
            create_distribution_figure(goals_data, 'Goals', LFCColors.PRIMARY_RED),
            create_distribution_figure(assists_data, 'Assists', LFCColors.SECONDARY_GREEN),
            ""
        )
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return (
            error_msg,
            create_distribution_figure(np.array([]), 'Goals', LFCColors.PRIMARY_RED),
            create_distribution_figure(np.array([]), 'Assists', LFCColors.SECONDARY_GREEN),
            error_msg
        )

# ---------------------------
# Debug Callback
# ---------------------------
@callback(
    Output('debug-store', 'data'),
    Input('num-games-slider', 'value')
)
def debug_callback(value):
    """Log critical debugging information"""
    try:
        print(f"Slider value: {value}")
        goals, assists = load_data()
        print(f"Data loaded - Goals: {len(goals)}, Assists: {len(assists)}")
        print(f"Sample goals: {goals[:5]}")
        return None
    except Exception as e:
        print(f"Debug error: {str(e)}")
        return None

# ---------------------------
# Server Configuration
# ---------------------------
if __name__ == '__main__':
    app.run_server(
        host=os.environ.get('HOST', '0.0.0.0'),
        port=int(os.environ.get('PORT', 8050)),
        debug=os.environ.get('DEBUG', 'false').lower() == 'true'
    )
