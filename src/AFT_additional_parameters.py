import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem

def track_additional_metrics(df, time_step): #take https://www.nature.com/articles/s41592-025-02935-5#MOESM1 as a base
    records = []
    df = df.sort_values(by=['TRACK_ID', 'FRAME']).copy()
    for i in df['TRACK_ID'].unique():
        df_temp = df.loc[df['TRACK_ID'] == i]
        dx = df_temp['POSITION_X'].shift(-1) - df_temp['POSITION_X'] #dx(t,τ)=x(t+τ)−x(t)
        dy = df_temp['POSITION_Y'].shift(-1) - df_temp['POSITION_Y'] #dy(t,τ)=y(t+τ)−y(t)
        dR = (dx, dy) #dR(t,τ)=(dx(t,τ),dy(t,τ))
        dR_eucledian = np.sqrt(dx**2 + dy**2) #||dR(t,τ)||=sqrt(dx(t,τ)^2+dy(t,τ)^2)
        average_traveled_distance = dR_eucledian.mean() #davg(𝜏)<||dR(t,τ)||>
        net_distance = np.sqrt((df_temp['POSITION_X'].iloc[-1] - df_temp['POSITION_X'].iloc[0])**2 +
                              (df_temp['POSITION_Y'].iloc[-1] - df_temp['POSITION_Y'].iloc[0])**2) #dnet​=​dR(t=0,τ=nmax​Δt)​
        total_distance_travelled = dR_eucledian.sum() #dtot(​τ)=​Σ||dR(t,τ)||​
        if total_distance_travelled != 0:
            consistency_index = net_distance / total_distance_travelled #CI=dnet​/dtot
        else:
            consistency_index = np.nan
        instantaneous_speed = dR_eucledian / time_step #vinst​(t,τ)=||dR(t,τ)||/Δt
        avg_instantaneous_speed = instantaneous_speed.mean() #s(τ)= <|d(τ)|> / τ
        #velocity_correlation_index = np.dot(dR, dR.shift(+1))/(dR_eucledian * dR_eucledian.shift(+1)) #VCI(t,τ)=dR(t,τ)⋅dR(t+τ,τ)/|dR(t,τ)|⋅|dR(t+τ,τ)| maybe do it in numpy format? 
        velocity_correlation_index = np.nanmean((dx*dx.shift(-1) + dy*dy.shift(-1)) /(dR_eucledian * dR_eucledian.shift(-1))) #VCI(t,τ)=dR(t,τ)⋅dR(t+τ,τ)/|dR(t,τ)|⋅|dR(t+τ,τ)|
        #angular displacement in AFT
        displacement_autocorrelation_function = np.nanmean((dx*dx.shift(-1) + dy*dy.shift(-1)))

        records.append({
                'TRACK_ID': i,
                'FRAME': df_temp['FRAME'],
                'dx': dx,
                'dy': dy,
                'dR': dR,
                'dR_eucledian': dR_eucledian,
                'average_traveled_distance': average_traveled_distance,
                'net_distance': net_distance,
                'total_distance_travelled': total_distance_travelled,
                'consistency_index': consistency_index,
                'instantaneous_speed': instantaneous_speed,
                'avg_instantaneous_speed': avg_instantaneous_speed,
                'velocity_correlation_index': velocity_correlation_index,
                'displacement_autocorrelation_function': displacement_autocorrelation_function
            })
        
        return pd.DataFrame(records)
    #do we have a situation when several points for one time frame (?)

'''
def MSD_per_track(df, lag):
    df = df.sort_values(by=['TRACK_ID', 'FRAME']).copy()
    for i in df['TRACK_ID'].unique():
        df_temp = df.loc[df['TRACK_ID'] == i]
        dx = df_temp['POSITION_X'].shift(-lag) - df_temp['POSITION_X'] #dx(t,τ)=x(t+τ)−x(t)
        dy = df_temp['POSITION_Y'].shift(-lag) - df_temp['POSITION_Y'] #dy(t,τ)=y(t+τ)−y(t)
        MSD = np.nanmean(dx**2 + dy**2)
'''

def track_additional_metrics_per_ECM(df, time_step): #take https://www.nature.com/articles/s41592-025-02935-5#MOESM1 as a base
    records = []
    df = df.sort_values(by=['TRACK_ID', 'FRAME', 'ECM_ID']).copy()
    for i in df['TRACK_ID'].unique():
        for j in df['ECM_ID'].unique():
            df_temp = df.loc[(df['TRACK_ID'] == i) & (df['ECM_ID'] == j)]
            dx = df_temp['POSITION_X'].shift(-1) - df_temp['POSITION_X'] #dx(t,τ)=x(t+τ)−x(t)
            dy = df_temp['POSITION_Y'].shift(-1) - df_temp['POSITION_Y'] #dy(t,τ)=y(t+τ)−y(t)
            dR = (dx, dy) #dR(t,τ)=(dx(t,τ),dy(t,τ))
            dR_eucledian = np.sqrt(dx**2 + dy**2) #||dR(t,τ)||=sqrt(dx(t,τ)^2+dy(t,τ)^2)
            average_traveled_distance = dR_eucledian.mean() #davg(𝜏)<||dR(t,τ)||>
            net_distance = np.sqrt((df_temp['POSITION_X'].iloc[-1] - df_temp['POSITION_X'].iloc[0])**2 +
                                (df_temp['POSITION_Y'].iloc[-1] - df_temp['POSITION_Y'].iloc[0])**2) #dnet​=​dR(t=0,τ=nmax​Δt)​
            total_distance_travelled = dR_eucledian.sum() #dtot(​τ)=​Σ||dR(t,τ)||​
            if total_distance_travelled != 0:
                consistency_index = net_distance / total_distance_travelled #CI=dnet​/dtot
            else:
                consistency_index = np.nan
            instantaneous_speed = dR_eucledian / time_step #vinst​(t,τ)=||dR(t,τ)||/Δt
            avg_instantaneous_speed = instantaneous_speed.mean() #s(τ)= <|d(τ)|> / τ
            #velocity_correlation_index = np.dot(dR, dR.shift(+1))/(dR_eucledian * dR_eucledian.shift(+1)) #VCI(t,τ)=dR(t,τ)⋅dR(t+τ,τ)/|dR(t,τ)|⋅|dR(t+τ,τ)| maybe do it in numpy format? 
            velocity_correlation_index = np.nanmean((dx*dx.shift(-1) + dy*dy.shift(-1)) /(dR_eucledian * dR_eucledian.shift(-1))) #VCI(t,τ)=dR(t,τ)⋅dR(t+τ,τ)/|dR(t,τ)|⋅|dR(t+τ,τ)|
            #angular displacement in AFT
            displacement_autocorrelation_function = np.nanmean((dx*dx.shift(-1) + dy*dy.shift(-1)))
#should we devide by the full lenght of the track? In one ECM we will have 2 points in another 8 points - how to compare then?
            records.append({
                    'TRACK_ID': i,
                    'ECM_ID': j,
                    'FRAME': df_temp['FRAME'],
                    'dx_ECM': dx,
                    'dy_ECM': dy,
                    'dR_ECM': dR,
                    'dR_eucledian_ECM': dR_eucledian,
                    'average_traveled_distance_ECM': average_traveled_distance,
                    'net_distance_ECM': net_distance,
                    'total_distance_travelled_ECM': total_distance_travelled,
                    'consistency_index_ECM': consistency_index,
                    'instantaneous_speed_ECM': instantaneous_speed,
                    'avg_instantaneous_speed_ECM': avg_instantaneous_speed,
                    'velocity_correlation_index_ECM': velocity_correlation_index,
                    'displacement_autocorrelation_function_ECM': displacement_autocorrelation_function
                })
        return pd.DataFrame(records)

def line_metrics(df, time_step):
    records = []
    df = df.sort_values(by=['TRACK_ID', 'FRAME']).copy()
    for i in df['TRACK_ID'].unique():
        df_temp = df.loc[(df['TRACK_ID'] == i)]

        dx = df_temp['POSITION_X'] - df_temp['POSITION_X'].shift()
        dy = df_temp['POSITION_Y'] - df_temp['POSITION_Y'].shift()
        dR_euclidean = np.sqrt(dx**2 + dy**2)
        v = dR_euclidean / time_step
        total_path_length = dR_euclidean.sum()
        
        #contribution of each point to the FMI 
        FMI_x_plus = dx[dx > 0] / total_path_length
        FMI_x_minus = abs(dx[dx < 0]) / total_path_length
        FMI_y_plus = dy[dy > 0] / total_path_length
        FMI_y_minus = abs(dy[dy < 0]) / total_path_length

        records.append({'TRACK_ID': i,
                        'FRAME': df_temp['FRAME'],
                        'dx_spot': dx,
                        'dy_spot': dy,
                        'dR_euclidean_spot': dR_euclidean,
                        'v_spot': v,
                        'total_path_length': total_path_length,
                        'local_FMI_contribution_x_plus_spot': FMI_x_plus,
                        'local_FMI_contribution_x_minus_spot': FMI_x_minus,
                        'local_FMI_contribution_y_plus_spot': FMI_y_plus,
                        'local_FMI_contribution_y_minus_spot': FMI_y_minus})

    return pd.DataFrame(records)

def calculate_fiber_orientation(df):
    '''
    Calculation in 0 to 1 range, 0 along vertical axis, 1 along horizontal axis
    '''
    df = df.copy()
    theta_module = np.abs(df['AFT_angle'].dropna().values)
    df['AFT_angle_normalized'] = (theta_module / (np.pi/2))  # Normalize angle to [0, 1] range

    #print mean and median per condition
    mean_summary = df.groupby('Condition')['AFT_angle_normalized'].mean()
    median_summary = df.groupby('Condition')['AFT_angle_normalized'].median()

    print(f"Median values of AFT per condition:\n")
    print(median_summary.round(3))
    print(f"Mean values of AFT orientation per condition:\n")
    print(median_summary.round(3))

    return df['AFT_angle_normalized']

def calculate_velocity(df, delta_t):

    records = []
    df = df.sort_values(by=['TRACK_ID', 'FRAME']).copy()
    df = df.sort_values(by=['TRACK_ID', 'Condition', 'File_name', 'FRAME']).copy()

    for i in df['TRACK_ID'].unique():
        df_temp = df.loc[df['TRACK_ID'] == i]

        if 'dx' in df_temp.columns or 'dy' in df_temp.columns:
            pass
        else:
            dx = df_temp['POSITION_X'].shift(-1) - df_temp['POSITION_X'] #dx(t,τ)=x(t+τ)−x(t)
            dy = df_temp['POSITION_Y'].shift(-1) - df_temp['POSITION_Y'] #dy(t,τ)=y(t+τ)−y(t)

        #instantaneous velocity components
        df['vx'] = df['dx'] / delta_t
        df['vy'] = df['dy'] / delta_t

        # Instantaneous speed
        df['instantaneous_velocity'] = np.sqrt(df['vx']**2 + df['vy']**2)

        # Velocity direction
        df['instantaneous_velocity_angle_rad'] = np.arctan2(df['vy'], df['vx'])
        df['instantaneous_velocity_angle_deg'] = np.degrees(df['instantaneous_velocity_angle_rad'])

        # Population average velocity per condition
        df['population_average_velocity'] = df.groupby('Condition')['instantaneous_velocity'].transform('mean')

    return df

def plot_instantaneous_velocity_polar(df, n_bins: 36):

    conditions = df['Condition'].unique()
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)

    for cond in conditions:
        angles = df.loc[df['Condition'] == cond, df['instantaneous_velocity_angle_rad']].dropna() % (2 * np.pi)
        counts, _ = np.histogram(angles, bins=bin_edges)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        bars = ax.bar(bin_centers, counts, width=2*np.pi/n_bins, bottom=0.0,
                      color='steelblue', edgecolor='black', alpha=0.7)

        ax.set_title(f'Directionality of track in each spot — Condition: {cond}')
        plt.show()
