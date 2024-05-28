import h5py
import numpy as np
import pandas as pd
import datetime
import os
import warnings  

# signal processing packages
from scipy.signal import decimate
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import sosfiltfilt

# plotting packages
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import proj3d # better aspect ratio
from matplotlib.animation import FuncAnimation, PillowWriter # For animations


# import some utitlity functions
from utils.utils import load_min_max_luna_data, load_preprocessed_das_data

# for debugging
# import pdb; pdb.set_trace()

## some general plotting conventions to be used here:
## - when plotting circumferential data, plot looking down, with cardinal north being '90' degrees
## - if possible, plot the wind direction and speed as well
## - when dealing with luna data, make sure to know how many nan's are being plotted
## - when plotting circumferential data, be careful on the direction of plotting (ccw or cw), will plot based on the 'x' data


### luna plotting functions
def plot_min_max_luna(directory_to_file, input_file_name,title,transparency=0.25,start_point = 0,strain_min = -20, strain_max=20):
    ''' Function to plot processed min, max data
    Args:
        directory_to_file:
        input_file_name:
        start_point: Rotate plotting 

    Returns:
    '''
    min_strain,max_strain= load_min_max_luna_data(directory_to_file = directory_to_file,
                                        input_file_name = input_file_name)
    
    # Generate x data
    n_bot_loop = np.shape(min_strain['bot-loop'])[0]
    n_top_loop = np.shape(min_strain['top-loop'])[0]

    # Convert start_point to radians and set the starting angle to 6 degrees clockwise
    start_point_rad = np.deg2rad(start_point)  # Convert start_point from degrees to radians
    clockwise_offset = -1  # To rotate clockwise, we need to negate the angles

    # Bottom loop
    rad_bot_loop = np.linspace(0, 2 * np.pi, n_bot_loop, endpoint=False)  # Trim ~3 inches of overlap
    rad_bot_loop = (rad_bot_loop * clockwise_offset + start_point_rad) % (2 * np.pi)

    # Top loop
    rad_top_loop = np.linspace(0, 2 * np.pi, n_top_loop, endpoint=False)  # Trim ~3 inches of overlap
    rad_top_loop = (rad_top_loop * clockwise_offset + start_point_rad) % (2 * np.pi)


    fig,axs = plt.subplots(1,2,subplot_kw={'projection':'polar'},figsize=(16,8),dpi=400)

    segments = ['top-loop', 'bot-loop']

    # loop through tower segments
    for segment in segments:
        if 'bot' in segment:
            axs[0].scatter(rad_bot_loop,min_strain[segment],
                        color='tab:blue',s = 10,zorder=1,alpha=transparency)
            axs[0].scatter(rad_bot_loop,max_strain[segment],
                        color='tab:red',s = 10,zorder=1,alpha=transparency)
        elif 'top' in segment:
            axs[1].scatter(rad_top_loop,min_strain[segment],
                        color='tab:blue',s = 10,zorder=1,alpha=transparency)
            axs[1].scatter(rad_top_loop,max_strain[segment],
                        color='tab:red',s = 10,zorder=1,alpha=transparency)
    
    line_indicies = [51,751,1589,1926,2560,3139,3685,4422]
    line_names = ['B','C','D','E','F','G','H','A']

    # adding lines to plots
    for index, name in zip(line_indicies,line_names):
        angle = rad_bot_loop[index]
        axs[0].axvline(x=angle,color='black',linestyle='--')
        # Add text slightly offset from the line
        # Adjust radius and alignment as needed
        axs[0].text(x=angle, y=strain_max*0.8,
        s= name, horizontalalignment='left', verticalalignment='center')

    
    axs[0].set_title('Lower Loop', fontsize=24)
    axs[1].set_title('Upper Loop', fontsize=24)
    axs[0].set_ylabel('micro Strain Envelope',fontsize=16,labelpad=-270,rotation=0)
    axs[1].set_ylabel('micro Strain Envelope',fontsize=16,labelpad=-270,rotation=0)
    axs[0].set_xlabel('',fontsize=16)
    axs[1].set_xlabel('',fontsize=16)
    axs[0].tick_params(axis='both', labelsize=16)
    axs[1].tick_params(axis='both', labelsize=16)
    axs[0].grid(True)
    axs[1].grid(True)
    axs[1].legend(fontsize=18,bbox_to_anchor=(1, 1), loc='upper left')
    legend_handles = [Line2D([0], [0], marker='o', color='tab:blue', markersize=10, linestyle='None'),
                      Line2D([0], [0], marker='o', color='tab:red', markersize=10, linestyle='None')]
    axs[1].legend(handles=legend_handles, labels=[r'$OFDR$ min', r'$OFDR$ max'], fontsize=18, loc='upper left', bbox_to_anchor=(1.05, 0.95)) # , r'$\phi - OTDR$ Envelopes' 
    axs[0].axis(ymin=strain_min,ymax=strain_max)
    axs[1].axis(ymin=strain_min,ymax=strain_max)
    fig.suptitle(title, fontsize=28)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.tight_layout()
    # plt.savefig('destination_path.eps', format='eps')
    plt.show()
    


def plot_min_max_diff_luna(directory_to_file, input_file_name, title, transparency=0.25, start_point=0, strain_min=-20, strain_max=20,wind_dir = None, wind_speed = None):
    ''' Function to plot processed min, max data and their difference
    Args:
        directory_to_file (str): Directory containing the file
        input_file_name (str): Name of the input file
        title (str): Title of the plot
        transparency (float): Transparency level of the scatter plot points
        start_point (int): Degree to start the plot from
        strain_min (int): Minimum strain value for the y-axis
        strain_max (int): Maximum strain value for the y-axis
        wind_dir (float): Wind direction in degrees from North
        wind_speed (float): Wind speed

    '''
    min_strain, max_strain = load_min_max_luna_data(directory_to_file=directory_to_file, input_file_name=input_file_name)
    
    # Generate x data
    n_bot_loop = np.shape(min_strain['bot-loop'])[0]
    n_top_loop = np.shape(min_strain['top-loop'])[0]

    # Convert start_point to radians and set the starting angle to 6 degrees clockwise
    start_point_rad = np.deg2rad(start_point)  # Convert start_point from degrees to radians
    clockwise_offset = -1  # To rotate clockwise, we need to negate the angles

    # Bottom loop
    rad_bot_loop = np.linspace(0, 2 * np.pi, n_bot_loop, endpoint=False)
    rad_bot_loop = (rad_bot_loop * clockwise_offset + start_point_rad) % (2 * np.pi)

    # Top loop
    rad_top_loop = np.linspace(0, 2 * np.pi, n_top_loop, endpoint=False)
    rad_top_loop = (rad_top_loop * clockwise_offset + start_point_rad) % (2 * np.pi)


    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(16, 8), dpi=400)

    # Calculate the difference between max and min strains
    strain_diff_bot = max_strain['bot-loop'] - min_strain['bot-loop']
    strain_diff_top = max_strain['top-loop'] - min_strain['top-loop']

    segments = [('bot-loop', rad_bot_loop, strain_diff_bot), ('top-loop', rad_top_loop, strain_diff_top)]

    # Loop through tower segments
    for i, (segment, rad_loop, strain_diff) in enumerate(segments):
        axs[i].scatter(rad_loop, strain_diff, color='tab:blue', label='Strain Difference', s = 10,zorder=1,alpha=transparency)  # Plot strain difference

    # Additional configuration
    for i, sub_title in enumerate(['Lower Loop', 'Upper Loop']):
        axs[i].set_title(label=sub_title, y=1.05,fontsize=24)
        axs[i].set_ylabel('micro Strain', fontsize=16, labelpad=-330, rotation=0)
        axs[i].tick_params(axis='both', labelsize=16)
        axs[i].grid(True)
        
    axs[1].legend(fontsize=18, loc='upper left', bbox_to_anchor=(1.05, 0.95)) # Only show the legend for the second plot
    


    if wind_dir is not None and wind_speed is not None:
        # Calculate the wind direction in radians and adjust for plot rotation
        wind_dir_rad = np.deg2rad(wind_dir) + np.deg2rad(start_point)
        wind_dir_rad = wind_dir_rad % (2 * np.pi)   # Normalize to ensure it is within 0 to 2π

        # Add wind direction arrow to each subplot
        for ax in axs:
            ax.annotate('', xy=(wind_dir_rad, strain_max * 0.9), xytext=(0, 0),
                        arrowprops=dict(facecolor='red', shrink=0, width=2, headwidth=8, alpha=0.5))
            ax.text(wind_dir_rad, strain_max*.95, f'{wind_speed} m/s', horizontalalignment='center', verticalalignment='bottom', fontsize=12, color='red')
        else:
            pass


    line_indicies = [51,751,1589,1926,2560,3139,3685,4422]
    line_names = ['B','C','D','E','F','G','H','A']

    # adding lines to plots
    for index, name in zip(line_indicies,line_names):
        angle = rad_bot_loop[index]
        axs[0].axvline(x=angle,color='black',linestyle='--')
        # Add text slightly offset from the line
        # Adjust radius and alignment as needed
        axs[0].text(x=angle, 
                    y=strain_max*0.8, 
                    s = name, 
                    horizontalalignment='left', 
                    verticalalignment='center')
    
    # Add cardinal direction labels
    labels = ['E', 'N', 'W', 'S']
    angles = np.deg2rad([0, 90, 180, 270])
    
    for label, angle in zip(labels, angles):
        axs[0].text(x=angle,
                    y= 1.08 * strain_max,
                    s= label, 
                    horizontalalignment='center', 
                    verticalalignment='center', 
                    fontsize=16)
        axs[1].text(x=angle,
                    y= 1.08 * strain_max,
                    s= label, 
                    horizontalalignment='center', 
                    verticalalignment='center', 
                    fontsize=16)

    # Custom ticks, excluding cardinal directions
    ticks = np.linspace(0, 2 * np.pi, 8, endpoint=False)  # Define all possible tick positions
    tick_positions = [tick for tick in ticks if not np.isclose(tick, angles).any()]  # Exclude cardinal angles

    # Apply custom tick positions
    axs[0].set_xticks(tick_positions)
    axs[1].set_xticks(tick_positions)



    axs[0].axis(ymin=strain_min, ymax=strain_max)
    axs[1].axis(ymin=strain_min, ymax=strain_max)
    fig.suptitle(title, fontsize=28)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()




# class strain_animation():
#     def __init__(self, x_data_c1, y_data_c1, x_data_c2=None, y_data_c2=None, x_data_c3=None, y_data_c3=None, x_data_c4=None, y_data_c4=None,strain_min=-100,strain_max=100,title='Dynamic Strain'):
#         # x_data is now considered dynamic, y_data_series is static and only one array
#         self.x_data_c1 = x_data_c1  # Static y data
#         self.y_data_c1 = y_data_c1  # Dynamic strain data

#         self.x_data_c2 = x_data_c2  # 
#         self.y_data_c2 = y_data_c2  # 
#         self.x_data_c3 = x_data_c3  # 
#         self.y_data_c3 = y_data_c3  # 
#         self.x_data_c4 = x_data_c4  # 
#         self.y_data_c4 = y_data_c4  # 

#         self.strain_min = strain_min
#         self.strain_max = strain_max
#         self.title = title
#         # Setup the figure and axes...
#         self.fig, self.axs = plt.subplots(2,2,figsize=(10,20))
#         # Then setup FuncAnimation.
#         self.ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.y_data_c1),
#                                            init_func=self.setup_plot, blit=True,interval=10) # interval is in milliseconds, our data is sampled at 100 Hz

#     def setup_plot(self):
#         """Initial drawing of the scatter plot."""
#         # Use the first set of x data for initial plot
#         # CHANNEL 1
#         self.scat_c1 = self.axs[0,0].scatter(self.y_data_c1[0], self.x_data_c1, s = 80, edgecolors='black',color='tab:red')
#         self.axs[0,0].axis([self.strain_min,self.strain_max, 0, 19])

#         # Initialize text annotation for time display
#         self.time_text = self.axs[0,1].text(0.92, 0.9, '', transform=self.axs[0,1].transAxes,
#                                       horizontalalignment='right', verticalalignment='top')
#         # CHANNEL 2
#         self.scat_c2 = self.axs[1,1].scatter(self.y_data_c2[0], self.x_data_c2, s = 80, edgecolors='black',color='tab:red')
#         self.axs[1,1].axis([self.strain_min,self.strain_max, 0, 19])

#         # CHANNEL 3
#         self.scat_c3 = self.axs[0,1].scatter(self.y_data_c3[0], self.x_data_c3, s = 80, edgecolors='black',color='tab:red')
#         self.axs[0,1].axis([self.strain_min,self.strain_max, 0, 19])

#         # CHANNEL 4
#         self.scat_c4 = self.axs[1,0].scatter(self.y_data_c4[0], self.x_data_c4, s = 80, edgecolors='black',color='tab:red')
#         self.axs[1,0].axis([self.strain_min,self.strain_max, 0, 19])

#         # Properties
#         self.axs[0,0].set_title('Channel 1', fontsize=24)
#         self.axs[0,1].set_title('Channel 3', fontsize=24)
#         self.axs[1,0].set_title('Channel 4', fontsize=24)
#         self.axs[1,1].set_title('Channel 2', fontsize=24)

#         self.axs[0,0].set_ylabel('Distance Along Axis of Turbine (m)',fontsize=16)
#         self.axs[1,0].set_ylabel('Distance Along Axis of Turbine (m)',fontsize=16)
#         self.axs[1,0].set_xlabel('micro Strain Envelope',fontsize=16)
#         self.axs[1,1].set_xlabel('micro Strain Envelope',fontsize=16)
#         self.axs[0,0].tick_params(axis='both', labelsize=16)
#         self.axs[1,0].tick_params(axis='both', labelsize=16)
#         self.axs[0,1].tick_params(axis='both', labelsize=16)
#         self.axs[1,1].tick_params(axis='both', labelsize=16)
#         self.axs[0,0].grid(True)
#         self.axs[0,0].grid(True)
#         self.axs[1,0].grid(True)
#         self.axs[0,1].grid(True)
#         self.axs[1,1].grid(True)
#         self.fig.suptitle(self.title, fontsize=28,y=0.98)

#         return self.scat_c1, self.scat_c2, self.scat_c3, self.scat_c4, self.time_text

#     def update(self, i):
#         """Update the scatter plot."""
#         # Update only x data since y is static in this setup  
#         self.scat_c1.set_offsets(np.column_stack((self.y_data_c1[i],self.x_data_c1)))

#         # Update the time text based on the current frame and sampling rate
#         current_time = i / 100
#         self.time_text.set_text(f'Time = {current_time:.2f} s')

#         # CHANNEL 2
#         self.scat_c2.set_offsets(np.column_stack((self.y_data_c2[i],self.x_data_c2)))
#         # CHANNEL 3
#         self.scat_c3.set_offsets(np.column_stack((self.y_data_c3[i],self.x_data_c3)))
#         # CHANNEL 4
#         self.scat_c4.set_offsets(np.column_stack((self.y_data_c4[i],self.x_data_c4)))

#         return self.scat_c1, self.scat_c2, self.scat_c3, self.scat_c4,self.time_text
    
#         # Optionally, adjust the x-axis limits dynamically based on the new x data
#         # self.ax.set_xlim([min(x), max(x)])

#     def save_animation(self, filename, fps=100):
#         """Save the animation to a file."""
#         # The writer to use - this example uses FFmpeg
#         Writer = animation.writers['ffmpeg']
#         writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
#         self.ani.save(filename, writer=writer)



def plot_min_max_luna_comparison(directory_to_file, list_of_input_file_name):
    ''' Function to plot several files of processed min, max data
    Args:
        directory_to_file: directory to file location
        list_of_input_file_name: list of file names to plot

    Returns:
    '''
    return



### das plotting functions
def plot_das_time_series(directory_to_file, input_file_name,channels,title,transparency=0.25,start_point = 0,strain_min = -20, strain_max=20):
    ''' Functino to plot time series data of das data
    Args:
        directory_to_file: directory to file location
        list_of_input_file_name: list of file names to plot
        channels: list of channel indicies

    Returns:
    '''
    # list of tower segments
    # segments = ['bot_a', 'mid_a', 'top_a',
    #             'bot_b', 'mid_b', 'top_b',
    #             'bot_c', 'mid_c', 'top_c',
    #             'bot_d', 'mid_d', 'top_d']
    segments = ['bot_a', 
                'bot_b', 
                'bot_c',
                'bot_d']
    
    strain,time_datetime,_ = load_preprocessed_das_data(directory_to_file = directory_to_file,
                               input_file_name = input_file_name)
    
    # plot segments
    fig,axs = plt.subplots(2,2,figsize=(30,10),dpi=400)

    for segment in segments:
        if '_a' in segment:
            print(np.size(time_datetime),np.shape(strain[segment]))
            axs[0,0].scatter(time_datetime,strain[segment][:,0],
                             color='tab:blue',s=10,zorder=1,alpha=transparency)
        elif '_b' in segment:
            axs[0,1].scatter(time_datetime,strain[segment][:,0],
                             color='tab:blue',s=10,zorder=1,alpha=transparency)
        elif '_c' in segment:
            axs[1,0].scatter(time_datetime,strain[segment][:,0],
                             color='tab:blue',s=10,zorder=1,alpha=transparency)
        elif '_d' in segment:
             axs[1,1].scatter(time_datetime,strain[segment][:,0],
                             color='tab:blue',s=10,zorder=1,alpha=transparency)
             
    
    # set up titles and formatting
    axs[0,0].set_title('Axis a', fontsize=24)
    axs[0,1].set_title('Axis b', fontsize=24)
    axs[1,0].set_title('Axis c', fontsize=24)
    axs[1,1].set_title('Axis d', fontsize=24)
    axs[0,0].set_ylabel('micro Strain Envelope',fontsize=16)
    axs[1,0].set_ylabel('micro Strain Envelope',fontsize=16)
    axs[1,0].set_xlabel('Time',fontsize=16)
    axs[1,1].set_xlabel('Time',fontsize=16)
    axs[0,0].tick_params(axis='both', labelsize=16)
    axs[1,0].tick_params(axis='both', labelsize=16)
    axs[1,0].grid(True)
    axs[0,0].grid(True)
    axs[1,1].grid(True)
    axs[0,1].grid(True)
    fig.suptitle(title, fontsize=28)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.tight_layout()
    plt.show()
    return





def plot_das_time_series_one_axis_3D(directory_to_file, input_file_name, axis='a', title='DAS Time Series', target_time=None, transparency=0.25, elev=30, azim=30, time_marker=None):
    '''
    Function to plot 3D time series data of DAS data for all indices within a specified axis.

    Args:
        directory_to_file: directory to file location
        input_file_name: file name to plot
        axis: specific axis to plot ('a', 'b', 'c', or 'd')
        title: title of the plot
        elev: Elevation angle in the z plane
        azim: Azimuthal angle in the x,y plane
        time_marker: list of datetimes for where we plot a plane with constant time
    '''
    # Load data
    strain, time_datetime, time = load_preprocessed_das_data(directory_to_file=directory_to_file,
                                                         input_file_name=input_file_name)


    # Get time indices for target times
    if target_time is None:
        pass
    else:
        # Initialize list to store indices
        target_time_indices = []
        for t in target_time:
            # Convert the string time to a datetime object, then to Unix time
            target_t = (pd.Timestamp(t) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1us')
            # Find the index of the closest time in the 'time' array
            index = np.abs(time - target_t).argmin()
            target_time_indices.append(index)
    
    # Convert datetime to Matplotlib float date format for plotting
    time_floats = mdates.date2num(time_datetime[target_time_indices[0]:target_time_indices[1]])
    
    bot_name = 'bot_' + axis
    mid_name = 'mid_' + axis
    top_name = 'top_' + axis

    # Calculate total height of each segment assuming 1 m spacing, each dataset is time x num_sensors
    bot_segment_height = strain[bot_name].shape[1]
    mid_segment_height = strain[mid_name].shape[1]
    top_segment_height = strain[top_name].shape[1]

    mid_offset = bot_segment_height + 2  # 2 meter gap between bottom and mid
    top_offset = mid_segment_height + mid_offset + 2  # 2 meter gap between mid and top

    # Create distance array
    distance_bot = np.arange(bot_segment_height)
    distance_mid = np.arange(mid_segment_height) + mid_offset
    distance_top = np.arange(top_segment_height) + top_offset
    total_distance = np.concatenate((distance_bot, distance_mid, distance_top))

    # Concatenating strain data
    total_strain = np.vstack([strain[bot_name].T, strain[mid_name].T, strain[top_name].T]) /10430.378350470453 # TODO: REMOVE AFTER FIXING BUG # total_strain (num_sensors x time)

    ## Creating 3D plot
    # fig = plt.figure(figsize=plt.figaspect(0.1)*10)
    fig = plt.figure(figsize=(20, 10),dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, y=.75)
    ax.set_xlabel('Time', labelpad=20)
    ax.set_ylabel('Distance (m)')
    ax.set_zlabel('Microstrain', labelpad=5)


    # plotting each sensor's time series
    for i in range(total_strain.shape[0]):
        y = np.full_like(time_floats, total_distance[i])  # Broadcast distance to match the size of time_floats
        ax.plot(time_floats, y, total_strain[i, target_time_indices[0]:target_time_indices[1]], color='tab:blue', alpha=transparency)



    if time_marker:
        time_marker_floats = mdates.date2num(pd.to_datetime(time_marker))
        for tm in time_marker_floats:
            X = np.full((2, 2), tm)
            Y, Z = np.meshgrid(np.linspace(min(total_distance), max(total_distance), 2), np.linspace(total_strain[:, target_time_indices[0]:target_time_indices[1]].min(), total_strain[:, target_time_indices[0]:target_time_indices[1]].max(), 2))
            ax.plot_surface(X, Y, Z, color='tab:red', alpha=0.75)
            # Add text label on the plane
            ax.text(tm, max(total_distance) * 1.15, total_strain[:, target_time_indices[0]:target_time_indices[1]].max() * 1.05, 'Brake event', color = 'black')

    # Setting the plotting limits
    ax.set_xlim(time_floats.min(), time_floats.max())
    ax.set_ylim(total_distance.min(), total_distance.max())
    ax.set_zlim(total_strain[:, target_time_indices[0]:target_time_indices[1]].min(), total_strain[:, target_time_indices[0]:target_time_indices[1]].max())

    # Setting the view angle
    ax.view_init(elev=elev, azim=azim)

    # Formatting the x-axis to display datetime
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.set_box_aspect(aspect=(4, 1.5, 1))
    # ax.set_box_aspect([4,1,1])
    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 2,0.5, 1]))


    # Now set color to white (or whatever is "invisible")
    # ax.xaxis.pane.set_edgecolor('w')
    # ax.yaxis.pane.set_edgecolor('w')
    # ax.zaxis.pane.set_edgecolor('w')

    #  Bonus: To get rid of the grid as well:
    # ax.grid(False)
    # plt.subplots_adjust(left=-0.05, right=1.05, top=1, bottom=0, wspace=0.05, hspace=-0.1)
    plt.tight_layout() # Adjust as needed
    # plt.savefig('test.png', bbox_inches='tight',pad_inches = 0, dpi = 300)
    plt.subplots_adjust(left=-.25, right=1.25, top=1.5, bottom=-.4)
    plt.show()
    





def plot_das_time_series_all_axis_3D(directory_to_file, input_file_name, title='DAS Time Series', target_time=None, transparency=0.25, elev=30, azim=30, time_marker=None, time_label='Event'):
    '''
    Function to plot 3D time series data of DAS data for all indices within each axis.

    Args:
        directory_to_file: directory to file location
        input_file_name: file name to plot
        title: title of the plot
        target_time: List of two datetimes for defining the time range
        transparency: Transparency level for the plot lines
        elev: Elevation angle in the z plane
        azim: Azimuthal angle in the x,y plane
        time_marker: List of datetime strings for where to plot a vertical plane
        time_label: Label to apply to the time marker plane
    '''
    # Load data
    strain, time_datetime, time = load_preprocessed_das_data(directory_to_file=directory_to_file,
                                                         input_file_name=input_file_name)

    if target_time:
        target_time_indices = [np.abs(time - ((pd.Timestamp(t) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1us'))).argmin() for t in target_time]
        time_floats = mdates.date2num(time_datetime[target_time_indices[0]:target_time_indices[1]])
    else:
        time_floats = mdates.date2num(time_datetime)

    # Initialize the distance and strain dictionary
    total_distance = {}
    total_strain = {}
    axes = ['a', 'b', 'c', 'd']

    # Preparing data for each axis
    for axis in axes:
        bot_name = 'bot_' + axis
        mid_name = 'mid_' + axis
        top_name = 'top_' + axis

        # Segment heights and offsets
        bot_segment_height = strain[bot_name].shape[1]
        mid_segment_height = strain[mid_name].shape[1]
        top_segment_height = strain[top_name].shape[1]

        mid_offset = bot_segment_height + 2  # 2 meter gap between bottom and mid
        top_offset = mid_segment_height + mid_offset + 2  # 2 meter gap between mid and top

        distance_bot = np.arange(bot_segment_height)
        distance_mid = np.arange(mid_segment_height) + mid_offset
        distance_top = np.arange(top_segment_height) + top_offset
        total_distance[axis] = np.concatenate((distance_bot, distance_mid, distance_top))
        total_strain[axis] = np.vstack([strain[bot_name].T, strain[mid_name].T, strain[top_name].T])/10430.378350470453 # TODO: REMOVE AFTER FIXING BUG!

    fig = plt.figure(figsize=(20, 10),dpi=300)
    
    time_marker_floats = mdates.date2num(pd.to_datetime(time_marker)) if time_marker else []

    # Creating 3D plot for each axis
    for i, axis in enumerate(axes):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.set_title( f'Axis {axis.upper()}', y=0.9)  # Adjust title spacing
        ax.set_xlabel('Time')
        ax.set_ylabel('Distance (m)')
        ax.set_zlabel('Microstrain')

        strain_plot = total_strain[axis][:, target_time_indices[0]:target_time_indices[1]] if target_time else total_strain[axis]
        distance_plot = total_distance[axis]

        # Plot each sensor's time series
        for j in range(strain_plot.shape[0]):
            y = np.full_like(time_floats, distance_plot[j])  # Broadcast distance to match the size of time_floats
            ax.plot(time_floats, y, strain_plot[j, :], color='tab:blue', alpha=transparency)

        # Plot time marker planes
        for tm in time_marker_floats:
            X = np.full((2, 2), tm)
            Y, Z = np.meshgrid(np.linspace(min(distance_plot), max(distance_plot), 2), np.linspace(strain_plot.min(), strain_plot.max(), 2))
            ax.plot_surface(X, Y, Z, color='tab:red', alpha=0.75)
            ax.text(tm, max(distance_plot), strain_plot.max(), time_label, color='black')

        # View angle
        ax.view_init(elev=elev, azim=azim)

        # Formatting the x-axis to display datetime
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.set_box_aspect(aspect=(2, 0.5, 0.5))
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    plt.tight_layout()
    fig.suptitle(title, fontsize=16,y=0.8)
    plt.subplots_adjust(left=-.25, right=1.25, top=3.75, bottom=-.4)
    # plt.subplot_tool()
    plt.show()


# def plot_das_time_series_all_axis_3D(directory_to_file, input_file_name, title='DAS Time Series', target_time=None, transparency=0.25, elev=30, azim=30, time_marker=None):
#     '''
#     Function to plot 3D time series data of DAS data for all indices within a specified axis.

#     Args:
#         directory_to_file: directory to file location
#         input_file_name: file name to plot
#         axis: specific axis to plot ('a', 'b', 'c', or 'd')
#         title: title of the plot
#         elev: Elevation angle in the z plane
#         azim: Azimuthal angle in the x,y plane
#         time_marker: list of datetimes for where we plot a plane with constant time
#     '''
#     # Load data
#     strain, time_datetime, time = load_preprocessed_das_data(directory_to_file=directory_to_file,
#                                                          input_file_name=input_file_name)

#     # Lists all axes
#     axes = ['a', 'b', 'c', 'd']

#     # Get time indices for target times
#     if target_time is None:
#         pass
#     else:
#         # Initialize list to store indices
#         target_time_indices = []
#         for t in target_time:
#             # Convert the string time to a datetime object, then to Unix time
#             target_t = (pd.Timestamp(t) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1us')
#             # Find the index of the closest time in the 'time' array
#             index = np.abs(time - target_t).argmin()
#             target_time_indices.append(index)
    
#     # Convert datetime to Matplotlib float date format for plotting
#     time_floats = mdates.date2num(time_datetime[target_time_indices[0]:target_time_indices[1]])

#     # Initialize the distance and strain dictionary
#     total_distance = {}
#     total_strain = {}

#     # Generate position data and format strain data for each axis
#     for i,axis in enumerate(axes):
#         bot_name = 'bot_' + axis
#         mid_name = 'mid_' + axis
#         top_name = 'top_' + axis

#         # Calculate total height of each segment assuming 1 m spacing, each dataset is time x num_sensors
#         bot_segment_height = strain[bot_name].shape[1]
#         mid_segment_height = strain[mid_name].shape[1]
#         top_segment_height = strain[top_name].shape[1]

#         mid_offset = bot_segment_height + 2  # 2 meter gap between bottom and mid
#         top_offset = mid_segment_height + mid_offset + 2  # 2 meter gap between mid and top

#         # Create distance array
#         distance_bot = np.arange(bot_segment_height)
#         distance_mid = np.arange(mid_segment_height) + mid_offset
#         distance_top = np.arange(top_segment_height) + top_offset
#         total_distance[axis] = np.concatenate((distance_bot, distance_mid, distance_top))

#         # Concatenating strain data
#         total_strain[axis] = np.vstack([strain[bot_name].T, strain[mid_name].T, strain[top_name].T])  # total_strain (num_sensors x time)

#     fig = plt.figure(figsize=(20, 20))

    
#     ## Creating 3D plot sub plot (2x2)
#     for i, axis in enumerate(axes):
#         ax = fig.add_subplot(2,2,i+1, projection='3d')
#         ax.set_title(title, y = .8)
#         ax.set_xlabel('Time')
#         ax.set_ylabel('Distance (m)')
#         ax.set_zlabel('Microstrain')

#         strain_plot = total_strain[axis][:, target_time_indices[0]:target_time_indices[1]]
#         distance_plot = total_distance[axis]

#         # plotting each sensor's time series
#         for j in range(strain_plot.shape[0]):
#             y = np.full_like(time_floats, distance_plot[j])  # Broadcast distance to match the size of time_floats
#             ax.plot(time_floats, y, strain_plot[j, :], color='tab:blue', alpha=transparency)


#         if time_marker:
#             time_marker_floats = mdates.date2num(pd.to_datetime(time_marker))
#             for tm in time_marker_floats:
#                 X = np.full((2, 2), tm)
#                 Y, Z = np.meshgrid(np.linspace(min(distance_plot), max(distance_plot), 2), np.linspace(strain_plot.min(), strain_plot.max(), 2))
#                 ax.plot_surface(X, Y, Z, color='tab:red', alpha=0.75)
#                 # Add text label on the plane
#                 label_position = (tm, max(distance_plot) * 0.95, strain_plot.max() * 0.95)  # Adjust label position as needed
#                 ax.text(tm, max(distance_plot) * 1.15, strain_plot.max() * 1.05, 'Brake event', color = 'black')

#         # Setting the plotting limits
#         ax.set_xlim(time_floats.min(), time_floats.max())
#         ax.set_ylim(distance_plot.min(), distance_plot.max())
#         ax.set_zlim(strain_plot.min(), strain_plot.max())

#         # Setting the view angle
#         ax.view_init(elev=elev, azim=azim)

#         # Formatting the x-axis to display datetime
#         ax.xaxis.set_major_locator(mdates.AutoDateLocator())
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
#         ax.set_box_aspect(aspect=(4, 1, 1))
#         # Get rid of colored axes planes
#         # First remove fill
#         ax.xaxis.pane.fill = False
#         ax.yaxis.pane.fill = False
#         ax.zaxis.pane.fill = False

#     # Now set color to white (or whatever is "invisible")
#     # ax.xaxis.pane.set_edgecolor('w')
#     # ax.yaxis.pane.set_edgecolor('w')
#     # ax.zaxis.pane.set_edgecolor('w')

#     #  Bonus: To get rid of the grid as well:
#     # ax.grid(False)
#     # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)
#     plt.tight_layout() # Adjust as needed
#     plt.show()







































# class strain_animation():
#     def __init__(self, x_data_c1, y_data_c1, x_data_c2=None, y_data_c2=None, x_data_c3=None, y_data_c3=None, x_data_c4=None, y_data_c4=None,strain_min=-100,strain_max=100,title='Dynamic Strain'):
#         # x_data is now considered dynamic, y_data_series is static and only one array
#         self.x_data_c1 = x_data_c1  # Static y data
#         self.y_data_c1 = y_data_c1  # Dynamic strain data

#         self.x_data_c2 = x_data_c2  # 
#         self.y_data_c2 = y_data_c2  # 
#         self.x_data_c3 = x_data_c3  # 
#         self.y_data_c3 = y_data_c3  # 
#         self.x_data_c4 = x_data_c4  # 
#         self.y_data_c4 = y_data_c4  # 

#         self.strain_min = strain_min
#         self.strain_max = strain_max
#         self.title = title
#         # Setup the figure and axes...
#         self.fig, self.axs = plt.subplots(2,2,figsize=(10,20))
#         # Then setup FuncAnimation.
#         self.ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.y_data_c1),
#                                            init_func=self.setup_plot, blit=True,interval=10) # interval is in milliseconds, our data is sampled at 100 Hz

#     def setup_plot(self):
#         """Initial drawing of the scatter plot."""
#         # Use the first set of x data for initial plot
#         # CHANNEL 1
#         self.scat_c1 = self.axs[0,0].scatter(self.y_data_c1[0], self.x_data_c1, s = 80, edgecolors='black',color='tab:red')
#         self.axs[0,0].axis([self.strain_min,self.strain_max, 0, 19])

#         # Initialize text annotation for time display
#         self.time_text = self.axs[0,1].text(0.92, 0.9, '', transform=self.axs[0,1].transAxes,
#                                       horizontalalignment='right', verticalalignment='top')
#         # CHANNEL 2
#         self.scat_c2 = self.axs[1,1].scatter(self.y_data_c2[0], self.x_data_c2, s = 80, edgecolors='black',color='tab:red')
#         self.axs[1,1].axis([self.strain_min,self.strain_max, 0, 19])

#         # CHANNEL 3
#         self.scat_c3 = self.axs[0,1].scatter(self.y_data_c3[0], self.x_data_c3, s = 80, edgecolors='black',color='tab:red')
#         self.axs[0,1].axis([self.strain_min,self.strain_max, 0, 19])

#         # CHANNEL 4
#         self.scat_c4 = self.axs[1,0].scatter(self.y_data_c4[0], self.x_data_c4, s = 80, edgecolors='black',color='tab:red')
#         self.axs[1,0].axis([self.strain_min,self.strain_max, 0, 19])

#         # Properties
#         self.axs[0,0].set_title('Channel 1', fontsize=24)
#         self.axs[0,1].set_title('Channel 3', fontsize=24)
#         self.axs[1,0].set_title('Channel 4', fontsize=24)
#         self.axs[1,1].set_title('Channel 2', fontsize=24)

#         self.axs[0,0].set_ylabel('Distance Along Axis of Turbine (m)',fontsize=16)
#         self.axs[1,0].set_ylabel('Distance Along Axis of Turbine (m)',fontsize=16)
#         self.axs[1,0].set_xlabel('micro Strain Envelope',fontsize=16)
#         self.axs[1,1].set_xlabel('micro Strain Envelope',fontsize=16)
#         self.axs[0,0].tick_params(axis='both', labelsize=16)
#         self.axs[1,0].tick_params(axis='both', labelsize=16)
#         self.axs[0,1].tick_params(axis='both', labelsize=16)
#         self.axs[1,1].tick_params(axis='both', labelsize=16)
#         self.axs[0,0].grid(True)
#         self.axs[0,0].grid(True)
#         self.axs[1,0].grid(True)
#         self.axs[0,1].grid(True)
#         self.axs[1,1].grid(True)
#         self.fig.suptitle(self.title, fontsize=28,y=0.98)

#         return self.scat_c1, self.scat_c2, self.scat_c3, self.scat_c4, self.time_text

#     def update(self, i):
#         """Update the scatter plot."""
#         # Update only x data since y is static in this setup  
#         self.scat_c1.set_offsets(np.column_stack((self.y_data_c1[i],self.x_data_c1)))

#         # Update the time text based on the current frame and sampling rate
#         current_time = i / 100
#         self.time_text.set_text(f'Time = {current_time:.2f} s')

#         # CHANNEL 2
#         self.scat_c2.set_offsets(np.column_stack((self.y_data_c2[i],self.x_data_c2)))
#         # CHANNEL 3
#         self.scat_c3.set_offsets(np.column_stack((self.y_data_c3[i],self.x_data_c3)))
#         # CHANNEL 4
#         self.scat_c4.set_offsets(np.column_stack((self.y_data_c4[i],self.x_data_c4)))

#         return self.scat_c1, self.scat_c2, self.scat_c3, self.scat_c4,self.time_text
    
#         # Optionally, adjust the x-axis limits dynamically based on the new x data
#         # self.ax.set_xlim([min(x), max(x)])

#     def save_animation(self, filename, fps=100):
#         """Save the animation to a file."""
#         # The writer to use - this example uses FFmpeg
#         Writer = animation.writers['ffmpeg']
#         writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
#         self.ani.save(filename, writer=writer)
