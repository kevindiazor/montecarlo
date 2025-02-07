from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os

# ---------------------------
# Data Loading Optimization
# ---------------------------
SEASON_FILTER = '2023-24'
NUM_SIMULATIONS = 10_000

# Load data with query pushdown
df = pd.read_csv("mo_salah.csv").query("SEASON == @SEASON_FILTER")
goals = df['G'].to_numpy()
assists = df['A'].to_numpy()

# ---------------------------
# Vectorized Simulation
# ---------------------------
def monte_carlo_simulation(num_games: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Monte Carlo simulation for goals and assists."""
    size = (NUM_SIMULATIONS, num_games)
    goals_sim = np.random.choice(goals, size=size).sum(axis=1)
    assists_sim = np.random.choice(assists, size=size).sum(axis=1)
    return goals_sim, assists_sim

# ---------------------------
# Dashboard Configuration
# ---------------------------
app = Dash(__name__, 
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           meta_tags=[{"viewport": "width=device-width, initial-scale=1, shrink-to-fit=no"}])

# ---------------------------
# Reusable Components
# ---------------------------
slider = dcc.Slider(
    id='num-games-slider',
    min=10,
    max=38,
    step=1,
    value=38,
    marks={i: str(i) for i in range(10, 39, 4)},
    tooltip={"placement": "bottom", "always_visible": True}
)

def create_histogram(data: np.ndarray, color: str, title: str) -> dict:
    """Generate consistent histogram configuration."""
    mean = data.mean()
    p5, p95 = np.percentile(data, [5, 95])
    
    return {
        'data': [{
            'x': data,
            'type': 'histogram',
            'marker': {'color': color},
            'hovertemplate': 'Goals: %{x}<extra></extra>'
        }],
        'layout': {
            'title': {
                'text': f"{title} Distribution<br>Mean: {mean:.1f}, 90% Range: {p5}-{p95}",
                'x': 0.5,
                'xanchor': 'center'
            },
            'margin': {'t': 60},
            'xaxis': {'title': f'Total {title}'},
            'yaxis': {'title': 'Frequency'},
            'plot_bgcolor': '#f8f9fa'
        }
    }

# ---------------------------
# App Layout
# ---------------------------
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Mohamed Salah Season Predictor", className="text-center my-4"))),
    
    dbc.Row(dbc.Col([
        html.P("Select remaining games to simulate:", className="lead"),
        slider
    ], className="mb-4")),
    
    dbc.Row(dbc.Col(html.Div(id='slider-output', className="h5 text-center mb-4"))),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='goals-distribution'), xs=12, md=6),
        dbc.Col(dcc.Graph(id='assists-distribution'), xs=12, md=6)
    ])
], fluid=True)

# ---------------------------
# Callback Optimization
# ---------------------------
@app.callback(
    [Output('slider-output', 'children'),
     Output('goals-distribution', 'figure'),
     Output('assists-distribution', 'figure')],
    Input('num-games-slider', 'value')
)
def update_output(num_games: int):
    goals_data, assists_data = monte_carlo_simulation(num_games)
    
    goals_fig = create_histogram(goals_data, '#1f77b4', 'Goals')
    assists_fig = create_histogram(assists_data, '#2ca02c', 'Assists')
    
    summary = f"Simulating {num_games} games: {goals_data.mean():.1f}±{(goals_data.std()):.1f} goals, " \
              f"{assists_data.mean():.1f}±{assists_data.std():.1f} assists expected"
    
    return summary, goals_fig, assists_fig

# ---------------------------
# Server Configuration
# ---------------------------
if __name__ == '__main__':
    app.run_server(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8050)),
        debug=os.environ.get("DEBUG", "False") == "True"
    )
