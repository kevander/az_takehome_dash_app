from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "24rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


# make dash app layout
def make_layout():
    layout = html.Div(children=[
        html.H1(children='AZ Take-Home Data Simulation', style={'textAlign':'center'}),
        dcc.Graph(id='graph-content',
                      style={
                        'width': '100%',  # Takes full width of container
                        'height': '900px'  # Sets a fixed height
                    }
                  ),
        html.Hr(),
        # add centered title of table here
        html.H2(children='Cox Time-Varying Proportional Hazard Results', style={'textAlign':'center'}),
        # add table here
        dag.AgGrid(
            id='cox-results',
            columnDefs=[],
            rowData=[],
            style={'height': '600px', 'width': '100%'},
            columnSize="sizeToFit",
            className="ag-theme-alpine"
        )
    ], style=CONTENT_STYLE)
    return layout

def make_sidebar():
    # the style arguments for the sidebar. We use position:fixed and a fixed width
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "22rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
        "overflow-y": "auto"
    }
    input_background_color = '#EAF2F8'

    group_a_inputs = dbc.Card([
                dbc.CardBody([
                    
                    # Number of the visits
                    dbc.Label(html.B("N"), style={"font-size": "20px"}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Patients"),
                            dbc.Input(type="number", min=0, max=1500, value=500, step=10, id='n-group-a', style={"backgroundColor": input_background_color}),
                            dbc.FormText("Sample Size"),
                        ]),
                        dbc.Col([
                            # Number of the visits
                            dbc.Label("Visits"),
                            dbc.Input(type="number", min=0, max=15, value=10, step=1, id="n-visits-a", style={"backgroundColor": input_background_color}),
                            dbc.FormText("Total Visits"),
                        ])
                    ]),
                    
                    html.Br(),
                    html.Br(),

                    # Number of the visits
                    dbc.Label(html.B("Event Rate"), style={"font-size": "20px"}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label(["Baseline Rate", html.Br(), "(0-1)"]),
                            dbc.Input(type="number", min=0, max=1, value=.01, step=0.001, id="event-rate-baseline-a", style={"backgroundColor": input_background_color}),
                        ]),
                        dbc.Col([
                            # Number of the visits
                            dbc.Label("Per Visit Change (-1-1)"),
                            dbc.Input(type="number", min=-1, max=1, value=.01, step=0.001, id="event-rate-change-a", style={"backgroundColor": input_background_color}),
                        ])
                    ]),
                    dbc.FormText("Set the baseline event rate (0-1) and the per-visit change (-1 - 1)."),

                    html.Br(),
                    html.Br(),

                    # Relationship Coefficient
                    dbc.Label(html.B("Relationship Coefficient"), style={"font-size": "20px"}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Relationship Coefficient"),
                            dbc.Input(type="number", min=-10, max=10, value=-0.05, step=0.01, id="rel-coef-a", style={"backgroundColor": input_background_color}),
                        ]),
                    ]),
                    dbc.FormText("Set the relationship coefficient."),

                    html.Br(),
                    html.Br(),

                    dbc.Label(html.B("Continuous Measure"), style={"font-size": "20px"}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Mean"),
                            dbc.Input(type="number", min=20, max=150, value=100, step=1, id="cv-mean-a", style={"backgroundColor": input_background_color}),
                        ]),
                        dbc.Col([
                            dbc.Label("St. Dev"),
                            dbc.Input(type="number", min=-100, max=100, value=15, step=1, id="cv-std-a", style={"backgroundColor": input_background_color}),
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Noise"),
                            dbc.Input(type="number", min=-4, max=4, value=1, step=.1, id="cv-noise-a", style={"backgroundColor": input_background_color}),
                        ]),
                        dbc.Col([
                            dbc.Label("Change (-5 to 5)"),
                            dbc.Input(type="number", min=-50, max=50, value=-2, step=1, id="cv-change-rate-a", style={"backgroundColor": input_background_color}),
                        ])
                    ]),
                ])
            ])

    group_b_inputs = dbc.Card([
                dbc.CardBody([
                    
                    # Number of the visits
                    dbc.Label(html.B("N"), style={"font-size": "20px"}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Patients"),
                            dbc.Input(type="number", min=0, max=1500, value=500, step=10, id='n-group-b', style={"backgroundColor": input_background_color}),
                            dbc.FormText("Sample Size"),
                        ]),
                        dbc.Col([
                            # Number of the visits
                            dbc.Label("Visits"),
                            dbc.Input(type="number", min=0, max=15, value=10, step=1, id="n-visits-b", style={"backgroundColor": input_background_color}),
                            dbc.FormText("Total Visits"),
                        ])
                    ]),
                    
                    html.Br(),
                    html.Br(),

                    # Number of the visits
                    dbc.Label(html.B("Event Rate"), style={"font-size": "20px"}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label(["Baseline Rate", html.Br(), "(0-1)"]),
                            dbc.Input(type="number", min=0, max=1, value=.01, step=0.001, id="event-rate-baseline-b", style={"backgroundColor": input_background_color}),
                        ]),
                        dbc.Col([
                            # Number of the visits
                            dbc.Label("Per Visit Change (-1-1)"),
                            dbc.Input(type="number", min=-1, max=1, value=.01, step=0.001, id="event-rate-change-b", style={"backgroundColor": input_background_color}),
                        ])
                    ]),
                    dbc.FormText("Set the baseline event rate (0-1) and the per-visit change (-1 - 1)."),

                    html.Br(),
                    html.Br(),

                    # Relationship Coefficient
                    dbc.Label(html.B("Relationship Coefficient"), style={"font-size": "20px"}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Relationship Coefficient"),
                            dbc.Input(type="number", min=-10, max=10, value=-0.01, step=0.001, id="rel-coef-b", style={"backgroundColor": input_background_color}),
                        ]),
                    ]),
                    dbc.FormText("Set the relationship coefficient."),

                    html.Br(),
                    html.Br(),

                    dbc.Label(html.B("Continuous Measure"), style={"font-size": "20px"}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Mean"),
                            dbc.Input(type="number", min=20, max=150, value=100, step=1, id="cv-mean-b", style={"backgroundColor": input_background_color}),
                        ]),
                        dbc.Col([
                            dbc.Label("St. Dev"),
                            dbc.Input(type="number", min=-100, max=100, value=15, step=1, id="cv-std-b", style={"backgroundColor": input_background_color}),
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Noise"),
                            dbc.Input(type="number", min=-4, max=4, value=.5, step=.1, id="cv-noise-b", style={"backgroundColor": input_background_color}),
                        ]),
                        dbc.Col([
                            dbc.Label("Change (-5 to 5)"),
                            dbc.Input(type="number", min=-50, max=50, value=-2, step=1, id="cv-change-rate-b", style={"backgroundColor": input_background_color}),
                        ])
                    ]),
                ])
            ])


    sidebar = html.Div(
        [
            html.H2("Simulation Parameters"),
            html.Hr(),

            dbc.Accordion(
                [
                    dbc.AccordionItem([
                        group_a_inputs
                    ], item_id='accordion-a', title=html.H4(html.B('Group A Parameters'))
                ), 
                dbc.AccordionItem([
                        group_b_inputs
                    ], item_id='accordion-b', title=html.H4(html.B('Group B Parameters'))
                ),
                ], active_item='accordion-b')
        ],
        style=SIDEBAR_STYLE,
    )
    return sidebar