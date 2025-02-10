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

# --- Mohamed Salah Data ---
CSV_FILE_SALAH = "mo_salah.csv"
df_salah = pd.read_csv(CSV_FILE_SALAH, usecols=['SEASON', 'G', 'A'])
df_salah = df_salah.loc[df_salah['SEASON'] == SEASON_FILTER]
goals = df_salah['G'].to_numpy()
assists = df_salah['A'].to_numpy()

# --- LeBron James Data ---
CSV_FILE_LEBRON = "statmuse (1).csv"
df_lebron = pd.read_csv(CSV_FILE_LEBRON)
PTS = df_lebron['PTS'].values
REB = df_lebron['REB'].values
AST = df_lebron['AST'].values

# Use modern random generator
rng = np.random.default_rng()

# ---------------------------
# Monte Carlo Simulation Functions
# ---------------------------
def monte_carlo_simulation_salah(num_games: int) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo simulation for Salah (Goals, Assists)."""
    size = (NUM_SIMULATIONS, num_games)
    goals_sim = rng.choice(goals, size=size).sum(axis=1)
    assists_sim = rng.choice(assists, size=size).sum(axis=1)
    return goals_sim, assists_sim

def monte_carlo_simulation_lebron(num_games: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte Carlo simulation for LeBron (PTS, AST, REB per game)."""
    size = (NUM_SIMULATIONS, num_games)
    simulated_pts = np.random.choice(PTS, size=size, replace=True).sum(axis=1) / num_games
    simulated_ast = np.random.choice(AST, size=size, replace=True).sum(axis=1) / num_games
    simulated_reb = np.random.choice(REB, size=size, replace=True).sum(axis=1) / num_games
    return simulated_pts, simulated_ast, simulated_reb

def create_histogram(data: np.ndarray, color: str, title: str) -> dict:
    """Generate Plotly histogram figure configuration."""
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

# ---------------------------
# App Initialization and Light/Dark Toggle
# ---------------------------
external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME]
app = Dash(__name__,
           external_stylesheets=external_stylesheets,
           meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1, shrink-to-fit=yes"}])

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
# Layout Components
# ---------------------------
# Salah simulation slider
salah_slider = dcc.Slider(
    id='num-games-slider',
    min=10,
    max=38,
    step=1,
    value=38,
    marks={i: str(i) for i in range(10, 39, 4)},
    tooltip={"placement": "bottom", "always_visible": True}
)

# LeBron simulation slider
lebron_slider = dcc.Slider(
    id='num-games-slider-lebron',
    min=60,
    max=82,
    step=1,
    value=82,
    marks={i: str(i) for i in range(60, 83, 2)},
    tooltip={"placement": "bottom", "always_visible": True}
)

# Main layout with Tabs for simulation selection
app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            dcc.Tabs(
                id='simulation-tabs',
                value='salah',  # default active tab
                children=[
                    dcc.Tab(label='Mohamed Salah', value='salah'),
                    dcc.Tab(label='LeBron James', value='lebron')
                ]
            ),
            width=12
        )
    ),
    html.Div(id='tabs-content'),
    dbc.Row(dbc.Col(color_mode_switch))
], fluid=True)

# ---------------------------
# Callback: Render Content Based on Active Tab
# ---------------------------
@app.callback(Output('tabs-content', 'children'),
              Input('simulation-tabs', 'value'))
def render_tab_content(active_tab):
    if active_tab == 'salah':
        return html.Div([
            html.H1(
                "Mohamed Salah Season Predictor",
                className="text-center my-4",
                style={'color': '#C8102E'}
            ),
            html.Img(src=app.get_asset_url("salah_image.jpg"),
                     style={"width": "300px", "display": "block", "margin": "0 auto"}),
            dbc.Row(
                dbc.Col([
                    html.P("Select remaining games to simulate:", className="lead"),
                    salah_slider
                ], className="mb-4")
            ),
            dbc.Row(
                dbc.Col(html.Div(id='slider-output', className="h5 text-center mb-4"))
            ),
            dbc.Row([
                dbc.Col(dcc.Graph(id='goals-distribution'), xs=12, md=6),
                dbc.Col(dcc.Graph(id='assists-distribution'), xs=12, md=6)
            ])
        ])
    elif active_tab == 'lebron':
        return html.Div([
            html.H1(
                "LeBron James Season Predictor",
                className="text-center my-4",
                style={'color': '#552583'}
            ),
            html.Img(src=app.get_asset_url("lebron_image.jpg"),
                     style={"width": "300px", "display": "block", "margin": "0 auto"}),
            dbc.Row(
                dbc.Col([
                    html.P("Select remaining games to simulate:", className="lead"),
                    lebron_slider
                ], className="mb-4")
            ),
            dbc.Row(
                dbc.Col(html.Div(id='slider-output-lebron', className="h5 text-center mb-4"))
            ),
            dbc.Row([
                dbc.Col(dcc.Graph(id='points-distribution'), xs=12, md=4),
                dbc.Col(dcc.Graph(id='assists-distribution-lebron'), xs=12, md=4),
                dbc.Col(dcc.Graph(id='rebounds-distribution'), xs=12, md=4)
            ])
        ])

# ---------------------------
# Callback: Update Mohamed Salah Graphs
# ---------------------------
@app.callback(
    [Output('slider-output', 'children'),
     Output('goals-distribution', 'figure'),
     Output('assists-distribution', 'figure')],
    Input('num-games-slider', 'value')
)
def update_salah_output(num_games: int):
    goals_data, assists_data = monte_carlo_simulation_salah(num_games)
    
    goals_fig = create_histogram(goals_data, '#C8102E', 'Goals')
    assists_fig = create_histogram(assists_data, '#00B2A9', 'Assists')
    
    goals_mean = goals_data.mean()
    goals_5th = np.percentile(goals_data, 5)
    goals_95th = np.percentile(goals_data, 95)
    
    summary_text = (
        f"Simulating {num_games} games, Salah is predicted to score between "
        f"{goals_5th} and {goals_95th} goals, with an average of {goals_mean:.2f} goals."
    )
    
    return summary_text, goals_fig, assists_fig

# ---------------------------
# Callback: Update LeBron James Graphs
# ---------------------------
@app.callback(
    [Output('slider-output-lebron', 'children'),
     Output('points-distribution', 'figure'),
     Output('assists-distribution-lebron', 'figure'),
     Output('rebounds-distribution', 'figure')],
    Input('num-games-slider-lebron', 'value')
)
def update_lebron_output(num_games: int):
    simulated_pts, simulated_ast, simulated_reb = monte_carlo_simulation_lebron(num_games)
    
    points_fig = create_histogram(simulated_pts, '#FDF667', 'PTS')
    assists_fig = create_histogram(simulated_ast, '#00B2A9', 'AST')
    rebounds_fig = create_histogram(simulated_reb, '#C8102E', 'REB')
    
    summary_text = (
        f"In a simulated {num_games}-game season, LeBron averaged {simulated_pts.mean():.2f} PTS, "
        f"{simulated_ast.mean():.2f} AST, and {simulated_reb.mean():.2f} REB per game."
    )
    
    return summary_text, points_fig, assists_fig, rebounds_fig

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
