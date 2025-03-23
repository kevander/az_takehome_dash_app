import IPython
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import ui
import simulation

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')

# FLATLY
# LUMEN
# YETI
app = Dash(external_stylesheets=[dbc.themes.LUMEN])
server = app.server

# Requires Dash 2.17.0 or later
content = ui.make_layout()
sidebar = ui.make_sidebar()

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

@callback(
    Output('graph-content', 'figure'),
    Output('cox-results', 'rowData'),
    Output('cox-results', 'columnDefs'),
    Input('n-group-a', 'value'),
    Input('n-visits-a', 'value'),
    Input('event-rate-baseline-a', 'value'),
    Input('event-rate-change-a', 'value'),
    Input('rel-coef-a', 'value'),
    Input("cv-mean-a", 'value'),
    Input("cv-std-a", 'value'),
    Input("cv-noise-a", 'value'),
    Input("cv-change-rate-a", 'value'),

    Input('n-group-b', 'value'),
    Input('n-visits-b', 'value'),
    Input('event-rate-baseline-b', 'value'),
    Input('event-rate-change-b', 'value'),
    Input('rel-coef-b', 'value'),
    Input("cv-mean-b", 'value'),
    Input("cv-std-b", 'value'),
    Input("cv-noise-b", 'value'),
    Input("cv-change-rate-b", 'value'),
)
def update_graph(n_a = 500,
                 n_visits_a = 10, 
                 event_rate_a = 0.05,
                 event_rate_change_a = 0.02,
                 relationship_coefficient_a = 0.05,
                 cv_mean_a = 100,
                 cv_sd_a = 20,
                 cv_noise_sd_a = 3,
                 cv_change_over_time_a = -3,
                 
                 n_b = 500,
                 n_visits_b = 10,
                 event_rate_b = 0.05,
                 event_rate_change_b = 0.02,
                 relationship_coefficient_b = 0.05,
                 cv_mean_b = 100,
                 cv_sd_b = 20,
                 cv_noise_sd_b = 3,
                 cv_change_over_time_b = -3
                 ):
    # Low Dependency Simulation 
    low_dep_sim = simulation.Simulation(sim_name='low_dependency',
                                        n=n_a,
                                        n_visits=n_visits_a,
                                        event_rate=event_rate_a,
                                        event_rate_change_rate=event_rate_change_a,
                                        cv_mean=cv_mean_a, 
                                        cv_sd=cv_sd_a,
                                        cv_change_over_time=cv_change_over_time_a,
                                        cv_noise_sd=cv_noise_sd_a, 
                                        relationship_coefficient=relationship_coefficient_a
                                        )
    low_dep_cv_data, low_dep_death_data = low_dep_sim.simulate()
    # High Dependency Simulation
    high_dep_sim = simulation.Simulation(sim_name='high_dependency',
                                        n=n_b,
                                        n_visits=n_visits_b,
                                        event_rate=event_rate_b,
                                        event_rate_change_rate=event_rate_change_b,
                                        cv_mean=cv_mean_b,
                                        cv_sd=cv_sd_b,
                                        cv_change_over_time=cv_change_over_time_b,
                                        cv_noise_sd=cv_noise_sd_b,
                                        relationship_coefficient=relationship_coefficient_b,
                                        patient_id_start_idx=n_a+1
                                        )
    high_dep_cv_data, high_dep_death_data = high_dep_sim.simulate()

    cv_plot_df = pd.concat([
        low_dep_cv_data,
        high_dep_cv_data
    ])

    event_plot_df = pd.concat([
        low_dep_death_data,
        high_dep_death_data
    ])

    fig = low_dep_sim.plot_continous_variable_over_time(cv_plot_df, event_plot_df)

    low_cox_res = low_dep_sim.run_cox_timevary_cox_propotional_hazard(low_dep_cv_data, low_dep_death_data)
    high_cox_res = high_dep_sim.run_cox_timevary_cox_propotional_hazard(high_dep_cv_data, high_dep_death_data)
    
    group_a_df = pd.DataFrame(low_cox_res.summary)
    group_b_df = pd.DataFrame(high_cox_res.summary)

    group_a_df.insert(0, 'Group', 'A')
    group_b_df.insert(0, 'Group', 'B')
    stats_df = pd.concat([group_a_df, group_b_df])

    stats_df = stats_df[['Group', 'coef', 'exp(coef)', 'se(coef)', 'z', 'p']]

    # Round
    stats_df['coef'] = stats_df['coef'].apply(lambda x: round(x, 3))
    stats_df['exp(coef)'] = stats_df['exp(coef)'].apply(lambda x: round(x, 3))
    stats_df['se(coef)'] = stats_df['se(coef)'].apply(lambda x: round(x, 3))
    stats_df['z'] = stats_df['z'].apply(lambda x: round(x, 3))
    stats_df['p'] = stats_df['p'].apply(lambda x: round(x,
                                                        8))

    columnDefs = [{'headerName': i, 'field': i} for i in stats_df.columns]
    rowData = stats_df.to_dict('records')

    return fig, rowData, columnDefs

if __name__ == '__main__':
    app.run(debug=True)
    