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
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# import some utitlity functions
from utils.utils import load_min_max_luna_data

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
    
    # generate x data
    n_bot_loop = np.shape(min_strain['bot-loop'])[0]
    n_top_loop = np.shape(min_strain['top-loop'])[0]

    
    # bottom loop
    rad_bot_loop = np.linspace(0, 2*np.pi, n_bot_loop, endpoint=False) # trim ~3 inches of overlap
    rad_bot_loop = (rad_bot_loop + (start_point*np.pi/180)) % (2*np.pi)
    # top loop
    rad_top_loop = np.linspace(0, 2*np.pi, n_top_loop, endpoint=False) # trim ~3 inches of overlap
    rad_top_loop = (rad_top_loop + (start_point*np.pi/180)) % (2*np.pi)

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
    



def plot_min_max_luna_comparison(directory_to_file, list_of_input_file_name):
    ''' Function to plot several files of processed min, max data
    Args:
        directory_to_file:
        list_of_input_file_name: list of file names to plot

    Returns:
    '''
    return


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