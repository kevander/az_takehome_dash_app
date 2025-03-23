import IPython
import numpy as np
import pandas as pd
import tqdm
from typing import List
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats
from lifelines import CoxTimeVaryingFitter
pio.renderers.default = "browser" 

class Simulation:
    '''
    Parameters
    ----------
    n : int
        Desired number of patients.

    n_visits: int
        Desired number of visits across the study.

    event_rate: float
        range: (0 - 1)
        Baseline event rate PER VISIT.
        For instance, if the event rate is 0.10, then 10% of patients will experience an event at each visit.

    event_rate_change_rate: float
        range: (0-1)
        The rate at which the event rate changes over time.
        For instance, if the event rate is 0.10 and the event_rate_change_rate is 0.01, then the event rate will increase by 1% each visit.

    cv_mean: float
        The mean of the continuous variable.

    cv_sd: float
        The standard deviation of the continuous variable.

    cv_change_over_time: float
        range: (-inf, inf)
        unit: CV standard deviations
        Set the rate of change of the continuous variable over time.
        The code will take this value and divide it by the number of visits to get the change at each visit.
        For instance, if cv_change_over_time is -2, then the continuous variable will decrease by 2 standard deviations over the course of the study.

    cv_noise_sd: float
        The standard deviation of the noise added to the continuous variable at each visit.
        If this is set to zero, then the CV for each patient will be a perfect linear trajectory.
        
    relationship_coefficient: float
        The strength of the relationship between the continuous variable and the event rate.

        If this is set to zero, then the event rate is independent of the continuous variable.
        If this is set to a positive value, then the event rate increases as the continuous variable increases.
        If this is set to a negative value, then the event rate decreases as the continuous variable increases.

        This directly maps to the beta coefficient in a logistic regression model that modulates the 
        relationship between the continuous variable and the event rate.
    
    patient_id_start_idx: int
        The starting index for patient IDs.
        This is useful if you want to simulate multiple arms and want to ensure that the patient IDs are unique across arms.

    seed: int
        The random seed for reproducibility.
    '''
    def __init__(self,  
                 sim_name: str, 
                 n: int = 500,
                 n_visits: int = 10,
                 event_rate: float = 0.05,
                 event_rate_change_rate: float = 0.01,
                 cv_mean: float = 100,
                 cv_sd: float = 20,
                 cv_noise_sd: float = 3,
                 cv_change_over_time: float = -2,
                 patient_id_start_idx: int = 1,
                 relationship_coefficient: float = .1,
                 seed: int = 2948,
                 ):
        self.sim_name = sim_name
        self.n = n
        self.n_visits = n_visits
        self.event_rate = event_rate
        self.event_rate_change_rate = event_rate_change_rate
        self.cv_mean = cv_mean
        self.cv_sd = cv_sd
        self.cv_noise_sd = cv_noise_sd
        self.cv_std_change_over_time = cv_change_over_time
        self.relationship_coefficient = relationship_coefficient
        self.seed = seed
        self.patient_id_start_idx = patient_id_start_idx

    def simulate(self):
        patient_ids = np.arange(self.patient_id_start_idx, 
                                self.patient_id_start_idx+self.n)
        cv_df, death_df = self.run_simulation(
            patient_ids=patient_ids,
            arm_name=self.sim_name,
            arm_n=self.n,
            event_rate=self.event_rate,
            event_rate_change_rate=self.event_rate_change_rate,
            cv_mean=self.cv_mean,
            cv_sd=self.cv_sd,
            cv_std_change_over_time=self.cv_std_change_over_time,
            cv_noise_sd=self.cv_noise_sd, 
            relationship_coefficient=self.relationship_coefficient
        )
        return cv_df, death_df

        # Run Placebo Arm Simulation 
    
    def run_simulation(self, 
                patient_ids: List[int],
                arm_name: str,
                arm_n: int,
                event_rate: float,
                event_rate_change_rate: float,
                cv_mean: float,
                cv_sd: float,
                cv_std_change_over_time: float,
                cv_noise_sd: float, 
                relationship_coefficient: float):
        
        # set seed
        np.random.seed(self.seed)
        
        # baseline continuous variable 
        cv_baseline = np.random.normal(loc=cv_mean, 
                                        scale=cv_sd, 
                                        size=arm_n)
        
        debug = False
        if debug:
            cv_std_change_over_time = 0
            event_rate = 0
            event_rate_change_rate = 0
            relationship_coefficient = 0

        #IPython.embed()
        ### C.V. Change Rates over Time
        # divide cv_std_change_over_time by n_visits to get the change at each visit
        # *cv_sd converts the change rate from SD to native units
        cv_change_at_each_visit = (cv_std_change_over_time / self.n_visits) * cv_sd

        # add some noise to the change rates, otherwise we have an unrealistic scenario
        # where all patients have the same trajectory
        cv_change_noise = np.random.normal(loc=0, 
                                        scale=abs(cv_change_at_each_visit*0.3), 
                                        size=arm_n)
        cv_change_rates = cv_change_at_each_visit + cv_change_noise

        visit_cv_data = []
        event_data = []
        for visit in range(self.n_visits):
            ### Simulate Continuous Variable Data
            # Baseline visit
            if visit == 0:
                visit_values = cv_baseline
            else: 
                event_rate += event_rate_change_rate
                # Get the C.V. values for this visit (rate x time)
                expected_values = cv_baseline + (cv_change_rates * visit)

                # Add noise to the trajectory
                noise_terms = np.random.normal(loc=0, scale=cv_noise_sd, size=arm_n)
                visit_values = expected_values + noise_terms
            visit_cv_data.append(pd.Series(visit_values))

            ### Simulate Events
            if visit == 0:
                events = np.zeros(arm_n)
            else:
                # sample from a bernoulli distribution to determine if the patient dies
                epsilon = 1e-10  # Small value to prevent division by zero
                event_rate_safe = np.clip(event_rate, epsilon, 1-epsilon)
                bias_term = np.log((1-event_rate_safe)/event_rate_safe) 
                event_prob = 1 / (1 + np.exp(bias_term + -relationship_coefficient * (visit_values - cv_mean) ))
                event_prob = np.clip(event_prob, 0, 1)  # Ensure probabilities are between 0 and 1

                events = np.random.binomial(n=1, p=event_prob, size=arm_n)
            event_data.append(pd.Series(events))

        # Concatenate CV/event data from each visit
        event_df    = pd.concat(event_data, axis=1)
        visit_cv_df = pd.concat(visit_cv_data, axis=1)

        # format the data and mask the CV measurements if the patient died
        cv_df, death_df = self.format_output(event_df, 
                                                visit_cv_df, 
                                                patient_ids, 
                                                format='long')
        # convert to ordered categorical
        death_df['visit'] = pd.Categorical(death_df['visit'],
                                            categories=range(1, self.n_visits+1),
                                            ordered=True)

        cv_df.insert(1, 'arm', arm_name)
        death_df.insert(1, 'arm', arm_name)
        return cv_df, death_df
    
    def format_output(self, 
                      event_df = pd.DataFrame, 
                      visit_cv_df = pd.DataFrame, 
                      patient_ids = List[int],
                      format='long'):
        if format == 'long':
            # Format continuous variable data
            cv_df = visit_cv_df.copy()
            cv_df.columns = range(1, self.n_visits+1)
            cv_df.insert(0, 'patient_id', patient_ids )
            visit_cv_long = cv_df.melt(id_vars='patient_id')
            visit_cv_long.columns = ['patient_id', 'visit', 'continuous_measure']

            # for each row, find first column with a 1
            # this corresponds to the visit where a death was observed
            death_visits = event_df.idxmax(axis=1)
            death_df = pd.DataFrame({
                'patient_id': patient_ids,
                'visit': death_visits
            })
            death_df['event'] = 0
            death_df.loc[death_df['visit'] != 0, 'event'] = 1
            death_df['visit'] = death_df['visit'].replace(0, self.n_visits)

            # remove cv measurements from after death
            filter_df = visit_cv_long.merge(death_df, 
                                on=['patient_id'], 
                                suffixes=('', '_death'),
                                how='left')
            filter_df.loc[filter_df['visit'] > filter_df['visit_death'], 'continuous_measure'] = np.nan
            filter_df.drop(columns=['visit_death', 'event'], inplace=True)
            filter_df = filter_df.sort_values(['patient_id', 'visit'])
            return filter_df, death_df

    def run_cox_timevary_cox_propotional_hazard(self,
                                                cv_data: pd.DataFrame,
                                                death_data: pd.DataFrame):
        # merge continuous and event data
        df = cv_data.merge(death_data, 
                            on=['patient_id', 'visit'], how='left')
        
        # drop visits that occured after death
        df = df.loc[df['continuous_measure'].notna()].copy()

        # if an event isn't recorded, then patient is alive
        df['event'] = df['event'].replace(np.nan, 0)

        # the lifelines package requires a 'start' column, indicating the start of the observation window
        df['start'] = df['visit'] - 1

        # The model also uses any extra columns as predictors, so drop any extra cols
        keep_cols = ['patient_id', 'visit', 'continuous_measure', 'event', 'start']
        cox_df = df[keep_cols].copy()
        cox_df['visit'] = cox_df['visit'].astype(int)

        # Fit the time-dependent Cox model
        ctv = CoxTimeVaryingFitter()
        ctv.fit(cox_df, id_col="patient_id", start_col="start", stop_col="visit", event_col="event")
        return ctv


    def plot_continous_variable_over_time(self, 
                                            cv_plot_df: pd.DataFrame,
                                            event_plot_df: pd.DataFrame):
        
        plot_df = cv_plot_df\
            .groupby(['arm', 'visit'])['continuous_measure']\
            .agg(['mean', 'std', 'sem']).reset_index()

        event_count_df = event_plot_df.groupby(['arm', 'visit'])['event'].sum().reset_index()
        event_count_df['cumsum'] = event_count_df.groupby('arm')['event'].cumsum()
        
        arm_count_df = event_plot_df['arm'].value_counts().reset_index()
        arm_count_df.columns = ['arm', 'n_patients']

        event_count_df = event_count_df.merge(arm_count_df, on='arm')
        event_count_df['event_perc'] = (100*(event_count_df['cumsum']/event_count_df['n_patients']))
        event_count_df['display_str'] = '<b>'+event_count_df['cumsum'].astype(str) + '</b><br>('+event_count_df['event_perc'].round(2).astype(str)+'%)'

        plot_df = plot_df.merge(event_count_df, on=['arm', 'visit'])

        # plotly scatter with errorbars and lines connecting each dot
        fig = go.Figure()

        color_list = [
            '#8A0051',
            '#EFAB00'
        ]
        for i,arm in enumerate(plot_df['arm'].unique()):
            color = color_list[i]
            arm_df = plot_df[plot_df['arm'] == arm]
            fig.add_trace(go.Scatter(
                x=arm_df['visit'],
                y=arm_df['mean'],
                mode='lines+markers',
                marker=dict(
                    size=15,
                    line=dict(width=2),
                    color=color
                ),
                line=dict(
                    width=2,
                    color=color
                ),
                name=arm,
                error_y=dict(
                    type='data',
                    array=arm_df['sem'],
                    visible=True,
                    thickness=4,
                    width=8,
                )
            ))
            # add event count annotations at bottom of the plot
            for j, row in arm_df.iterrows():
                if i != 0:
                    offset = 0.03
                else:
                    offset = 0.08

                fig.add_annotation(
                    x=row['visit'],
                    y=offset,
                    text=row['display_str'],
                    yref='paper',
                    showarrow=False,
                    yshift=-20,
                    font=dict(
                        size=14,
                        color=color
                    )
                )
        
        fig.add_annotation(
            x=1.14,
            y=0.03,
            yref='paper',
            xref='paper',
            text='N Events (%)',
            showarrow=False,
            font=dict(
                size=14,
                color='black'
            )
        )

        # make sure that each visit gets a tick
        fig.update_xaxes(tickmode='array', tickvals=plot_df['visit'].unique())

        # add black x and y axis lines
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=False)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=False)
        fig.update_layout(
            xaxis_title='Visit',
            yaxis_title='Continuous Measurement',
            showlegend=True
        )
        # add axis tick lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_layout(
            xaxis=dict(
                ticks="outside",  # Show tick marks
                ticklen=5,       # Length of the tick marks
                tickwidth=1,      # Width of the tick marks
                tickcolor="black" # Color of the tick marks
            ),
            yaxis=dict(
                ticks="outside",   # Tick marks inside the plot
                ticklen=5,        # Length of the tick marks
                tickwidth=1,    # Width of the tick marks
                tickcolor="black"   # Color of the tick marks
            )
        )
        # make font larger
        fig.update_layout(
            font=dict(
                size=22,
            )
        )

        return fig
