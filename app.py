from dash import Dash, dcc, html, Input, Output, callback, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os
from functools import lru_cache
from typing import Tuple

# ---------------------------
# Optimized Configuration
# ---------------------------
class Config:
    SEASON_FILTER = os.environ.get('SEASON_FILTER', '2023-24')
    NUM_SIMULATIONS = int(os.environ.get('NUM_SIMULATIONS', 5000))  # Increased for accuracy
    DEFAULT_GAMES = int(os.environ.get('DEFAULT_GAMES', 38))
    SALAH_IMAGE = "assets/salah_image.webp"  # WebP format for faster loading
    CACHE_TTL = 3600  # 1 hour cache

# ---------------------------
# Vectorized Data Loading
# ---------------------------
@lru_cache(maxsize=1)
def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load optimized parquet data with memory mapping"""
    base_dir = os.path.dirname(__file__)
    parquet_path = os.path.join(base_dir, 'mo_salah.parquet')  # Converted from CSV
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Data file not found at: {parquet_path}")

    df = pd.read_parquet(
        parquet_path,
        columns=['SEASON', 'G', 'A'],
        filters=[('SEASON', '==', Config.SEASON_FILTER)]
    )
    
    return df['G'].values, df['A'].values

# ---------------------------
# Optimized Simulation
# ---------------------------
class SimulationEngine:
    _rng = np.random.default_rng()
    _goals, _assists = load_data()
    
    @staticmethod
    @lru_cache(maxsize=Config.DEFAULT_GAMES)
    def monte_carlo(num_games: int) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized simulation with pre-allocated arrays"""
        idx = SimulationEngine._rng.integers(
            len(SimulationEngine._goals), 
            size=(Config.NUM_SIMULATIONS, num_games)
        )
        
        goals_sim = SimulationEngine._goals[idx].sum(axis=1)
        assists_sim = SimulationEngine._assists[idx].sum(axis=1)
        
        return goals_sim, assists_sim

# ---------------------------
# Memoized Figure Creation
# ---------------------------
class FigureFactory:
    _template = {
        'layout': {
            'margin': {'t': 80, 'b': 40, 'l': 40, 'r': 40},
            'plot_bgcolor': LFCColors.WHITE,
            'paper_bgcolor': LFCColors.WHITE,
            'font': {'color': LFCColors.DARK_RED}
        }
    }
    
    @staticmethod
    @lru_cache(maxsize=128)
    def create_figure(data: tuple, title: str, color: str) -> dict:
        """Cached figure generation with precomputed stats"""
        arr = np.array(data)
        if arr.size == 0:
            return FigureFactory._empty_figure(title)
            
        mean, std = arr.mean(), arr.std()
        p5, p95 = np.percentile(arr, [5, 95])
        
        return {
            'data': [{
                'x': arr,
                'type': 'histogram',
                'marker': {'color': color},
                'opacity': 0.85,
                'histnorm': 'probability density',
                'hovertemplate': f'{title}: %{{x:.1f}}<extra></extra>'
            }],
            'layout': {
                **FigureFactory._template['layout'],
                'title': f'<b>{title}</b> Distribution<br>μ={mean:.1f} σ={std:.1f} (90% CI: {p5}-{p95})',
                'xaxis': {'title': f'Total {title}', 'gridcolor': LFCColors.WHITE},
                'yaxis': {'title': 'Probability Density', 'gridcolor': LFCColors.WHITE}
            }
        }
    
    @staticmethod
    def _empty_figure(title: str) -> dict:
        return {
            'data': [],
            'layout': {
                'title': f'No {title} Data Available',
                'annotations': [{
                    'text': 'Data Not Loaded',
                    'showarrow': False,
                    'font': {'size': 18, 'color': LFCColors.PRIMARY_RED}
                }]
            }
        }

# ---------------------------
# Optimized Callbacks
# ---------------------------
@callback(
    [Output('slider-output', 'children'),
     Output('goals-distribution', 'figure'),
     Output('assists-distribution', 'figure'),
     Output('hidden-error', 'children')],
    Input('num-games-slider', 'value')
)
def update_components(num_games: int):
    """Optimized callback with batched processing"""
    try:
        goals_data, assists_data = SimulationEngine.monte_carlo(num_games)
        figures = (
            FigureFactory.create_figure(tuple(goals_data), 'Goals', LFCColors.PRIMARY_RED),
            FigureFactory.create_figure(tuple(assists_data), 'Assists', LFCColors.SECONDARY_GREEN)
        )
        
        return (
            f"Projection for {num_games} games: "
            f"{goals_data.mean():.1f}±{goals_data.std():.1f} goals | "
            f"{assists_data.mean():.1f}±{assists_data.std():.1f} assists",
            *figures,
            ""
        )
    except Exception as e:
        return handle_error(e)

def handle_error(e: Exception) -> tuple:
    error_msg = f"Error: {str(e)}"
    return (
        error_msg,
        FigureFactory.create_figure((), 'Goals', LFCColors.PRIMARY_RED),
        FigureFactory.create_figure((), 'Assists', LFCColors.SECONDARY_GREEN),
        error_msg
    )
