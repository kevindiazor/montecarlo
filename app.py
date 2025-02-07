from dash import Dash, dcc, html, Input, Output, clientside_callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os

# ---------------------------
# Constants and Data Loading
# ---------------------------
SEASON_FILTER = '2023-24'
NUM_SIMULATIONS = 10_000
CSV_FILE = "mo_salah.csv"

# Read only required columns and filter by season
df = pd.read_csv(CSV_FILE, usecols=['SEASON', 'G', 'A'])
df = df.loc[df['SEASON'] == SEASON_FILTER]
goals = df['G'].to_numpy()
assists = df['A'].to_numpy()

# Use modern random generator
rng = np.random.default_rng()

# ---------------------------
# Monte Carlo Simulation Function
# ---------------------------
def monte_carlo_simulation(num_games: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Monte Carlo simulation for goals and assists."""
    size = (NUM_SIMULATIONS, num_games)
    goals_sim = rng.choice(goals, size=size).sum(axis=1)
    assists_sim = rng.choice(assists, size=size).sum(axis=1)
    return goals_sim, assists_sim

# ---------------------------
# App Initialization and Light/Dark Toggle
# ---------------------------
external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME]
app = Dash(__name__, external_stylesheets=external_stylesheets,
           meta_tags=[{"viewport": "width=device-width, initial-scale=1, shrink-to-fit=no"}])

# Light/Dark mode switch component
color_mode_switch = html.Span(
    [
        dbc.Label(className="fa fa-moon", html_for="theme-switch"),
        dbc.Switch(
            id="theme-switch",
            value=True,   # True for light mode, False for dark mode
            persistence=True,
            className="d-inline-block ms-1",
        ),
        dbc.Label(className="fa fa-sun", html_for="theme-switch"),
    ],
    className="d-flex align-items-center mb-3",
)

# ---------------------------
# Dashboard Layout Components
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
            'hovertemplate': f'{title}: %{{x}}<extra></extra>'
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

app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H1(
                "Mohamed Salah Season Predictor",
                className="text-center my-4",
                style={'color': '#C8102E'}  # Liverpool red for the title
            )
        )
    ),
    # New row for Salah image
    dbc.Row(
        dbc.Col(
            html.Img(src=app.get_asset_url("salah_image.jpg"),
                     style={"width": "300px", "display": "block", "margin": "0 auto"})
        )
    ),
    dbc.Row(dbc.Col(color_mode_switch)),
    dbc.Row(
        dbc.Col([
            html.P("Select remaining games to simulate:", className="lead"),
            slider
        ], className="mb-4")
    ),
    dbc.Row(
        dbc.Col(html.Div(id='slider-output', className="h5 text-center mb-4"))
    ),
    dbc.Row([
        dbc.Col(dcc.Graph(id='goals-distribution'), xs=12, md=6),
        dbc.Col(dcc.Graph(id='assists-distribution'), xs=12, md=6)
    ])
], fluid=True)

# ---------------------------
# Callback: Update Graphs and Summary Text
# ---------------------------
@app.callback(
    [Output('slider-output', 'children'),
     Output('goals-distribution', 'figure'),
     Output('assists-distribution', 'figure')],
    Input('num-games-slider', 'value')
)
def update_output(num_games: int):
    goals_data, assists_data = monte_carlo_simulation(num_games)
    
    # Use Liverpool FC colors:
    # Goals histogram in Liverpool red (#C8102E)
    # Assists histogram in Liverpool green (#00B2A9)
    goals_fig = create_histogram(goals_data, '#C8102E', 'Goals')
    assists_fig = create_histogram(assists_data, '#00B2A9', 'Assists')
    
    # Calculate descriptive statistics for goals
    goals_mean = goals_data.mean()
    goals_5th = np.percentile(goals_data, 5)
    goals_95th = np.percentile(goals_data, 95)
    
    # Updated summary text including requested details
    summary_text = (
        f"Simulating {num_games} games per season, our model predicts Salah would score between "
        f"{goals_5th} and {goals_95th} goals, on average around {goals_mean:.2f} goals."
    )
    
    return summary_text, goals_fig, assists_fig

# ---------------------------
# Clientside Callback for Light/Dark Mode
# ---------------------------
clientside_callback(
    """
    function(switchOn) {
        if (switchOn) {
            document.documentElement.setAttribute("data-bs-theme", "light");
        } else {
            document.documentElement.setAttribute("data-bs-theme", "dark");
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("theme-switch", "id"),
    Input("theme-switch", "value")
)

# ---------------------------
# Run Server
# ---------------------------
if __name__ == '__main__':
    app.run_server(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8050)),
        debug=os.environ.get("DEBUG", "False") == "True"
    )
