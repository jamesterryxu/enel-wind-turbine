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

#import pdb; pdb.set_trace() Useful for debugging!

### das functions

# preprocessing functions

def raw_to_phase(directory_to_file,name_of_file):
    ''' Function to reformat the data fields in the h5 file (Doesn't decimate or unwrap)
    Args:
        directory_to_file: full directory to file (string)
        name_of_file: .h5 file that you want to decimate (string)
    Returns:

    Raises:
    '''
    # Open the HDF5 file
    file = h5py.File(directory_to_file+'/'+name_of_file+'.h5', 'r')
    # Read the dataset
    raw_data = file['/Acquisition/Raw[0]/RawData']
    raw_data = np.double(raw_data) # Convert to double
    # Transpose data
    raw_data = raw_data.T
    nch, _ = raw_data.shape
    # Import time
    raw_time = np.double(file['/Acquisition/Raw[0]/RawDataTime'])

    # Convert raw_data to phase_data
    phase_data = raw_data / 10430.378350470453 # This is (2**15)/pi
    
    # Save decimated strain data
    with h5py.File(directory_to_file+'/'+name_of_file+'_phase'+'.h5', 'w') as hf:
        hf.create_dataset('phase',  data=phase_data)
        hf.create_dataset('time',data=raw_time)
        hf.close()

def decim_to_100(directory_to_file,name_of_file,decim_factor=50):
    ''' Function to decimate the raw files into 100 Hz, and convert phase data to microstrain data
    Args:
        directory_to_file: full directory to file (string)
        name_of_file: .h5 file that you want to decimate (string)
        decim_factor: factor to decimate

    Returns:
        Saves a decimated .h5 file, with time and strain

    Raises:
    '''
    # Open the HDF5 file
    file = h5py.File(directory_to_file+'/'+name_of_file+'.h5', 'r')
    # Read the dataset
    raw_data = file['/Acquisition/Raw[0]/RawData']
    raw_data = np.double(raw_data) # Convert to double
    # Transpose data
    raw_data = raw_data.T
    nch, _ = raw_data.shape
    # Import time
    raw_time = np.double(file['/Acquisition/Raw[0]/RawDataTime'])

    # Convert raw_data to phase_data
    phase_data = raw_data / 10430.378350470453 # This is (2**15)/pi

    ### FILES ARE TOO BIG TO DO UNWRAPPING... for now skip this step
    # # UNWRAP PHASE_DECIM (RADIAN NUMBER) BEFORE DECIMATION 
    # phase_data_unwrapped = np.unwrap(phase_data,axis=1) # axis on time dimension

    # Decimate
    # 5000 Hz -> 100 Hz, decim_factor = 50
    # Get length of decimated vector
    # decim_length = len(decimate(phase_data_unwrapped[0,:], decim_factor))
    decim_length = len(decimate(phase_data[0,:], decim_factor))
    # Initialize phase_decim array
    # Use empty to make sure that we don't impute values
    phase_data_unwrapped_decim = np.empty((nch, decim_length))
    time_decim = np.empty((decim_length))

    # decim
    # Decimate each channel time series 
    for i in range(nch): # range(nch)
        phase_data_unwrapped_decim[i, :] = decimate(raw_data[i,:], decim_factor)

    # Decimate TIME data DON'T use decimate function! The decimate function downsamples the signal
    # after applying an anti-aliasing filter! Just take every decim_factor (50th) entry
    time_decim = raw_time[0::decim_factor]

    # Check if it is actually 100 Hz
    time_difference = time_decim[1] - time_decim[0]
    if (time_difference - 10000) < 1e-3: # datetime format here, 10000 is 0.01 seconds, unix format
        pass
    else:
        print(time_difference -10000)
        raise ValueError
    
    # Convert to strain
    ### Does this change with different channel readouts? CHECK!
    Lambd = 1550e-9 # wavelength for Rayleigh incident light, 1550 nm
    # if nch == 102:
    #     print(8)
    #     Lgauge = 8.167619
    # else:
    #     print(1)
    #     Lgauge = 2.0419046
    Lgauge = 2.0419046 #NOTE: Change for odh3 data! Lgauge = 8.167619
    n_FRI = 1.468200 # fiber refractive index
    PSF = 0.78 # photoelastic scaling factor xi

    strain_decim = (Lambd / (4*np.pi*n_FRI*Lgauge*PSF)) * phase_data_unwrapped_decim * 1e6 # microstrain
    
    # Save decimated strain data
    with h5py.File(directory_to_file+'/'+name_of_file+'_decimated100hz'+'.h5', 'w') as hf:
        hf.create_dataset('strain',  data=strain_decim)
        hf.create_dataset('time',data=time_decim)
        hf.close()

def batch_raw_to_phase(directory):
    ''' Convert to phase data of all .h5 files in the specified directory.
    Args:
        directory: The path to the directory containing the .h5 files.

    Returns:
        None. Phase files are saved in the same directory with the '_phase' suffix.
    '''
    # List all files in the directory
    files = os.listdir(directory)

    # Filter only .h5 files which do not contain 'phase' or 'decimated' in their filename
    h5_files = [f for f in files if f.endswith('.h5') and 'phase' not in f and 'decimated' not in f]

    # Decimate each of these files
    for file in h5_files:
        print(f"Converting {file} to phase data")
        # Extract the file name without the extension
        name_without_ext = os.path.splitext(file)[0]
        raw_to_phase(directory, name_without_ext)
        print(f"{file} done")

def batch_decim_to_100(directory):
    ''' Decimate all .h5 files in the specified directory.
    Args:
        directory: The path to the directory containing the .h5 files.

    Returns:
        None. Decimated files are saved in the same directory with the '_decimated100hz' suffix.
    '''
    # List all files in the directory
    files = os.listdir(directory)

    # Filter only .h5 files which do not contain 'decimated' in their filename
    h5_files = [f for f in files if f.endswith('.h5') and 'decimated' not in f]

    # Decimate each of these files
    for file in h5_files:
        print(f"Decimating {file}")
        # Extract the file name without the extension
        name_without_ext = os.path.splitext(file)[0]
        decim_to_100(directory, name_without_ext)
        print(f"{file} decimated")

def load_phase_data(directory_to_file,name_of_file):
    ''' Function to load decimated 100 Hz and process the datetimes
    Args:
        directory_to_file: full directory to file (string)
        name_of_file: .h5 file that you want to decimate (string)

    Returns:
        phase_data: numpy double of strain data
        time: list of datetimes

    Raises:
    '''
    file = h5py.File(directory_to_file+'/'+name_of_file+'.h5', 'r+')
    data = file['phase']
    time = file['time']
    # Convert decimated time data to datetime
    time_datetime = [datetime.datetime.fromtimestamp(i/1000000) for i in time]
    # convert h5 group to double
    return np.double(data),time_datetime

def load_decim_data(directory_to_file,name_of_file):
    ''' Main function to load decimated 100 Hz das data. Function also processes the datetimes
    Args:
        directory_to_file: full directory to file (string)
        name_of_file: .h5 file that you want to decimate (string)

    Returns:
        strain_data: numpy double of strain data
        time: list of datetimes

    Raises:
    '''
    file = h5py.File(directory_to_file+'/'+name_of_file+'.h5', 'r+')
    data = file['strain']
    time = file['time']
    # Convert decimated time data to datetime
    # time_datetime = [datetime.datetime.fromtimestamp(i/1000000) for i in time]
    # convert h5 group to double
    return np.double(data), time #time_datetime

def concatenate_and_save_h5(directory_to_file, output_filename):
    ''' Function to compile the decimated files, stitching all the files together
    '''
    files = [f for f in os.listdir(directory_to_file) if f.endswith('_decimated100hz.h5')]
    sorted_files = sort_filenames_by_time(files)

    # Read the first file to determine the shape of the data
    first_strain, _ = load_decim_data_helper(directory_to_file, sorted_files[0].replace('.h5', ''))
    num_spatial_points, _ = first_strain.shape

    # Initialize the arrays based on the shape of the first file
    all_strain_data = np.empty((num_spatial_points, 0))
    all_time_data = np.empty((0,))  # Assuming time data is 1D

    for file in sorted_files:
        print(file)
        strain_data, time_data = load_decim_data_helper(directory_to_file, file.replace('.h5', ''))
        all_strain_data = np.append(all_strain_data, strain_data, axis=1)
        all_time_data = np.append(all_time_data, time_data)  # Assuming time data is 1D

    # print(np.shape(all_strain_data))
    # print(np.shape(all_time_data))

    # Get correct file name to save, get the first datetime of the filenames
    filename = '_'.join(sorted_files[0].split('_')[:2])

    # Save data into a new h5 file
    with h5py.File(directory_to_file+'/'+filename+output_filename+'.h5', 'w') as f:
        f.create_dataset('strain', data=all_strain_data)
        f.create_dataset('time', data=all_time_data)

def clean_das_files_odh4(directory_to_file,input_file_name,output_file_name,target_time):
    ''' File to cut decimated das file from ODH4. 
        NOTE: We do flipping in this function. The end goal is to have a dataset that is ready to plot or do analysis with efficiently, with not much more pre-processing work
    Input:
        directory_to_file: directory to the folder where the input file lives
        input_file_name: name of file to be cut
        output_file_name: name of cut file (should follow naming convention)
        target_time: list of start and end times, takes a string in the format ('%Y-%m-%d %H:%M:%S') 

    Output:
        .h5 file that contains the cut data, with three fields
    '''
    file = h5py.File(directory_to_file+'/'+input_file_name+'.h5', 'r+')
    strain = np.double(file['strain']).T # all channels (in microstrain)
    # Need to transpose to make time x sensor!
    time = np.double(file['time']) # unix time



    # Initialize list to store indices
    target_time_indices = []
    for t in target_time:
        # Convert the string time to a datetime object, then to Unix time
        target_t = (pd.Timestamp(t) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1us')
        # Find the index of the closest time in the 'time' array
        index = np.abs(time - target_t).argmin()
        target_time_indices.append(index)

    # NOTE: These indexings are for the ODH4 data! Make sure to change when analyzing the odh3 data.
    # 4 axes, axis a is closest to door, b is natural progression of cable, CW looking up the tower, etc.
    # NOTE: We need to reverse the b and d axis
    # NOTE: TODO: Figure out if we need to reverse the loops or not... it's very confusing, need to check against photos
    # There are 3 tower segments, bot, mid, top (closest to nacelle)
    # 12 longitudinal 'segments', each with 2 indicies, e.g. [start_bot_a, end_bot_a]
    # axis a TODO: Check splicing rules, I think it's exclusive...
    bot_a = [41,61]
    mid_a = [61,88]
    top_a = [88,115]
    # axis b
    bot_b = [172,192]
    mid_b = [146,172]
    top_b = [119,146]
    # axis c
    bot_c = [198,218]
    mid_c = [218,244]
    top_c = [244,271]
    # axis d
    bot_d = [328,348]
    mid_d = [303,328]
    top_d = [275,303]


    # Save cut das data
    with h5py.File(directory_to_file + '/' + output_file_name + '.h5', 'w') as hf:
        hf.create_dataset('time', data = time[target_time_indices[0]:target_time_indices[1]])
        hf.create_dataset('bot_a', data = strain[target_time_indices[0]:target_time_indices[1],bot_a[0]:bot_a[1]])
        hf.create_dataset('mid_a', data = strain[target_time_indices[0]:target_time_indices[1],mid_a[0]:mid_a[1]])
        hf.create_dataset('top_a', data = strain[target_time_indices[0]:target_time_indices[1],top_a[0]:top_a[1]])
        hf.create_dataset('bot_b', data = np.flip(strain[target_time_indices[0]:target_time_indices[1],bot_b[0]:bot_b[1]]))
        hf.create_dataset('mid_b', data = np.flip(strain[target_time_indices[0]:target_time_indices[1],mid_b[0]:mid_b[1]]))
        hf.create_dataset('top_b', data = np.flip(strain[target_time_indices[0]:target_time_indices[1],top_b[0]:top_b[1]]))
        hf.create_dataset('bot_c', data = strain[target_time_indices[0]:target_time_indices[1],bot_c[0]:bot_c[1]])
        hf.create_dataset('mid_c', data = strain[target_time_indices[0]:target_time_indices[1],mid_c[0]:mid_c[1]])
        hf.create_dataset('top_c', data = strain[target_time_indices[0]:target_time_indices[1],top_c[0]:top_c[1]])
        hf.create_dataset('bot_d', data = np.flip(strain[target_time_indices[0]:target_time_indices[1],bot_d[0]:bot_d[1]]))
        hf.create_dataset('mid_d', data = np.flip(strain[target_time_indices[0]:target_time_indices[1],mid_d[0]:mid_d[1]]))
        hf.create_dataset('top_d', data = np.flip(strain[target_time_indices[0]:target_time_indices[1],top_d[0]:top_d[1]]))
        hf.close()
    return



# das helper functions
def sort_filenames_by_time(filenames):
    ''' Helper function to sort filenames
    '''
    return sorted(filenames, key=lambda x: datetime.datetime.strptime(x.split('_')[1], "%Y-%m-%dT%H%M%S%z"))

def load_decim_data_helper(directory_to_file,name_of_file):
    ''' Helper function to load decimated 100 Hz
    Args:
        directory_to_file: full directory to file (string)
        name_of_file: .h5 file that you want to decimate (string)
        decim_factor: factor to decimate

    Returns:
        strain_data: 
        time: 

    Raises:
    '''
    file = h5py.File(directory_to_file+'/'+name_of_file+'.h5', 'r+')
    data = file['strain']
    time = file['time']
    return data,time

## analysis functions
def load_preprocessed_das_data(directory_to_file,input_file_name):
    ''' Function to load decimated 100 Hz cut and process the datetimes
    Args:
        directory_to_file: full directory to file (string)
        input_file_name: .h5 file that you want to decimate (string)

    Returns:
        strain: dictionary of different segments containing numpy double of strain data
        time: list of datetimes

    Raises:
    '''
    # list of tower segments
    segments = ['bot_a', 'mid_a', 'top_a',
                'bot_b', 'mid_b', 'top_b',
                'bot_c', 'mid_c', 'top_c',
                'bot_d', 'mid_d', 'top_d']
    
    file = h5py.File(directory_to_file+'/'+input_file_name+'.h5', 'r+')

    # Initialize dictionary to store all data
    strain = {}
    for segment in segments:
        strain[segment] = np.double(file[segment])


    time = file['time']
    # Convert decimated time data to datetime
    time_datetime = [datetime.datetime.fromtimestamp(i/1000000) for i in time]
    # convert h5 group to double
    return strain,time_datetime,time


def filter_das_data(directory_to_file,input_file_name,cutoff_freq=0.1,order=2):
    ''' Function to filter data using a high pass filter
    Args:
        directory_to_file: full directory to file (string)
        input_file_name: .h5 file that you want to decimate (string)

    Returns:
        strain: dictionary of different segments containing numpy double of strain data that are now filtered
        time: list of datetimes

    Raises:
    '''
    # list of tower segments
    segments = ['bot_a', 'mid_a', 'top_a',
                'bot_b', 'mid_b', 'top_b',
                'bot_c', 'mid_c', 'top_c',
                'bot_d', 'mid_d', 'top_d']

    # load in data
    strain, _,time = load_preprocessed_das_data(directory_to_file=directory_to_file,
                               input_file_name=input_file_name)
    
    # get the sampling rate
    # need to make sure that the sampling rate is uniform
    time_diffs = np.diff(time)
    if not np.all(time_diffs == time_diffs[0]):
        warnings.warn('Time intervals are not uniform.', UserWarning)

    sampling_freq = (1 / time_diffs[0] )*1000000 # should be 100 Hz, unix time causing having to multiply by 1000000?
    print(sampling_freq)


    # getting the nyquist frequency and calculating the critical frequencies used for filter
    nyquist_freq = 0.5 * sampling_freq
    critical_freq = cutoff_freq / nyquist_freq

    high_pass_filter = butter(N=order,
                              Wn = critical_freq,
                              btype='highpass',
                              output = 'sos')
    
    # initialize the filtered dataset
    strain_filtered = {}

    # filter al data, loop through the segments
    for segment in segments:
        strain_filtered[segment] = sosfiltfilt(sos = high_pass_filter, 
                                      x = strain[segment],
                                      axis=0)
    
    # Save cut das data
    with h5py.File(directory_to_file + '/' + input_file_name + '_filtered.h5', 'w') as hf:
        hf.create_dataset('time', data = time)
        for segment in segments:
            hf.create_dataset(segment,data=strain_filtered[segment])

        hf.close()
    return


def min_max_das_data(directory_to_file,input_file_name):
    ''' Function to get the minimum and maximum of preprocessed das data
    Args:
        directory_to_file: full directory to file (string)
        input_file_name: .h5 file that you want to decimate (string)

    Returns:
        strain: dictionary of different segments containing numpy double of strain data that are now filtered
        time: list of datetimes

    Raises:
    '''
    # list of tower segments NOTE: Should eventually put this in a class or make it a global variable?
    segments = ['bot_a', 'mid_a', 'top_a',
                'bot_b', 'mid_b', 'top_b',
                'bot_c', 'mid_c', 'top_c',
                'bot_d', 'mid_d', 'top_d']
    
    # load in data
    strain, _,time = load_preprocessed_das_data(directory_to_file=directory_to_file,
                               input_file_name=input_file_name)
    
    # initialize min and max dictionaries
    min_strain = {}
    max_strain = {}

    for segment in segments:
        min_strain[segment] = np.nanmin(a = strain[segment],
                                        axis = 0)
        max_strain[segment] = np.nanmax(a = strain[segment],
                                        axis = 0)
    
    # Save the data arrays
    with h5py.File(directory_to_file + '/' + input_file_name + '_envelopes.h5', 'w') as hf:
        hf.create_dataset('time', data = time)
        for segment in segments:
            hf.create_dataset(segment+'min_strain',data=min_strain[segment])
            hf.create_dataset(segment+'max_strain',data=max_strain[segment])
        hf.close()

    return

# Helper function to load min and max strain
def load_min_max_das_data(directory_to_file,input_file_name):
    ''' Function to load min and max strain envelopes and process the datetimes
    Args:
        directory_to_file: full directory to file (string)
        input_file_name: .h5 file that you want to decimate (string)

    Returns:
        strain: dictionary of different segments containing numpy double of strain data
        time: list of datetimes

    Raises:
    '''
    # list of tower segments
    segments = ['bot_a', 'mid_a', 'top_a',
                'bot_b', 'mid_b', 'top_b',
                'bot_c', 'mid_c', 'top_c',
                'bot_d', 'mid_d', 'top_d']
    
    file = h5py.File(directory_to_file+'/'+input_file_name+'.h5', 'r+')

    # Initialize dictionary to store all data
    min_strain = {}
    max_strain = {}

    for segment in segments:
        min_strain[segment] = np.double(file[segment+'min_strain'])
        max_strain[segment] = np.double(file[segment+'max_strain'])


    time = file['time']
    # Convert decimated time data to datetime
    time_datetime = [datetime.datetime.fromtimestamp(i/1000000) for i in time]
    # convert h5 group to double
    return min_strain,max_strain,time_datetime,time



# Use fdd from package...




### luna functions
## preprocessing functions
def clean_luna_files(directory_to_file,input_file_name,output_file_name,target_time):
    ''' File to cut raw Luna file
    Input:
        directory_to_file: directory to the folder where the input file lives
        input_file_name: name of file to be cut
        output_file_name: name of cut file (should follow naming convention)
        target_time: list of start and end times, takes a string in the format ('%Y-%m-%d %H:%M:%S') Note that the luna data is 6 hours ahead of the local Oklahoma time!

    Output:
        .h5 file that contains the cut data, with three fields
    
    '''
    # Use pandas to help cut the data
    luna_data = pd.read_csv(directory_to_file+'/'+input_file_name,sep='\t',skiprows=32)

    # Get spatial indices to cut the dataset
    start_top_loop = luna_data.columns.get_loc('75.7894.1')
    end_top_loop = luna_data.columns.get_loc('87.5726.1')
    start_bot_loop = luna_data.columns.get_loc('36.3708.1')
    end_bot_loop = luna_data.columns.get_loc('49.7738.1')

    # Get time indices to cut the dataset
    # Change format to get indices
    luna_data.iloc[:,0] = pd.to_datetime(luna_data.iloc[:,0], format='%Y-%m-%d %H:%M:%S.%f')
    # Initialize list to store indices
    target_time_indices = []
    for time in target_time:
        time_diff = (luna_data.iloc[:,0]-pd.Timestamp(time)).abs()
        target_time_indices.append(time_diff.idxmin())

    # Now need to change to unix format (microseconds) to save as .h5 file. .h5 doesn't support datetimes so we'll have to convert when we load the data
    # Computing the unix time https://stackoverflow.com/questions/54313463/pandas-datetime-to-unix-timestamp-seconds
    luna_data.iloc[:,0] = (luna_data.iloc[:,0] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1us')

    # We now have all the indices needed to cut the dataset

    # Save cut luna data
    with h5py.File(directory_to_file+'/'+output_file_name+'.h5', 'w') as hf:
        hf.create_dataset('time',  data=luna_data.iloc[target_time_indices[0]:target_time_indices[1],0].values)
        hf.create_dataset('top-loop',data=luna_data.iloc[target_time_indices[0]:target_time_indices[1],start_top_loop:end_top_loop].values)
        hf.create_dataset('bot-loop',data=luna_data.iloc[target_time_indices[0]:target_time_indices[1],start_bot_loop:end_bot_loop].values)
        hf.close()
    return


def load_decim_luna_data(directory_to_file,name_of_file):
    ''' Main function to load cut luna data. Function also processes the datetimes
    Args:
        directory_to_file: full directory to file (string)
        name_of_file: .h5 file that you want to decimate (string)

    Returns:
        strain_data: numpy double of strain data
        time: list of datetimes

    Raises:
    '''
    file = h5py.File(directory_to_file+'/'+name_of_file+'.h5', 'r+')
    top_loop = file['top-loop']
    bot_loop = file['bot-loop']
    time = file['time']
    # Convert decimated time data to datetime
    time_datetime = [datetime.datetime.fromtimestamp(i/1000000) for i in time]
    # convert h5 group to double
    return np.double(top_loop),np.double(bot_loop),time_datetime

def datum_retriever_luna_data(directory_to_file,input_file_name,number_of_seconds=60):
    ''' Function to get zeroed out data for each bolt condition (Used for the brake dataset, last 5 minutes average)
    Args:
        directory_to_file: full directory to file (string)
        name_of_file: .h5 file that you want to decimate (string) (Luna data is sampled at 5 Hz)

    Returns:
        datum: numpy array of spatial points for each loop, for each bolt configuration
    '''
    file = h5py.File(directory_to_file+'/'+input_file_name+'.h5', 'r+')
    top_loop = file['top-loop']
    bot_loop = file['bot-loop']
    time = file['time']

    # Get index to average over
    sample_freq = 5 # 5 Hz
    number_of_samples = number_of_seconds * sample_freq

    # Check time is ok compared to number of samples
    if number_of_samples > len(time):
        raise ValueError('Requested number of seconds is too large')

    # Take the average over the last number_of_seconds
    top_loop_datum = np.mean(top_loop[-number_of_samples:],
                             axis=0)
    bot_loop_datum = np.mean(bot_loop[-number_of_samples:],
                             axis=0)
    
    # Get the min and max to confirm function works
    # Get the range (min and max) to confirm function works
    top_loop_range = np.max(top_loop[-number_of_samples:], axis=0) - np.min(top_loop[-number_of_samples:], axis=0)
    bot_loop_range = np.max(bot_loop[-number_of_samples:], axis=0) - np.min(bot_loop[-number_of_samples:], axis=0)

    # Check if the difference between min and max is greater than 15 microstrain for any sensor
    if np.any(top_loop_range > 15) or np.any(bot_loop_range > 15):
        warnings.warn('One or more sensors have a microstrain range greater than 15 over the specified period.', UserWarning)

    return top_loop_datum, bot_loop_datum


def zero_out_function_luna_data(directory_to_file,input_file_name,number_of_seconds=60):
    ''' Function to save another .h5 file that is zeroed out based on bolt configuration, using the datam of the last ~1 minute of data from the brake test data
        Note that we have to think a bit more carefully about the zeroing of the datasets that have bci-bcj data, what should we zero out by?
    Args:
        directory_to_file:
        name_of_file:

    Returns:
        Saves an .h5 file that is zeroed accordingly
    
    '''
    # load data
    file = h5py.File(directory_to_file+'/'+input_file_name+'.h5', 'r+')
    top_loop = file['top-loop']
    bot_loop = file['bot-loop']
    time = file['time']

    # get bolt configuration based on input_file_name
    bolt_configuration = input_file_name[-3:]

    top_loop_datum, bot_loop_datum = datum_retriever_luna_data(directory_to_file=directory_to_file,
                                                     input_file_name='brake-'+bolt_configuration,
                                                     number_of_seconds=number_of_seconds)
    
    # subtract datum for all time steps
    # Note that the datum is just one value for every spatial point, we want to subtract this per spatial point for every point in the time series
    top_loop_zeroed = top_loop - top_loop_datum
    bot_loop_zeroed = bot_loop - bot_loop_datum

    # Save zeroed data
    with h5py.File(directory_to_file+'/'+input_file_name+'_zeroed.h5', 'w') as hf:
        hf.create_dataset('time',  data=time)
        hf.create_dataset('top-loop',data= top_loop_zeroed )
        hf.create_dataset('bot-loop',data= bot_loop_zeroed )
        hf.close()
    return

    


# analysis functions
def load_preprocessed_luna_data(directory_to_file,input_file_name):
    ''' Function to load cut luna data and process the datetimes
    Args:
        directory_to_file: full directory to file (string)
        input_file_name: .h5 file that you want to decimate (string)

    Returns:
        strain: dictionary of different segments containing numpy double of strain data
        time: list of datetimes

    Raises:
    '''
    # list of strain segments
    segments = ['top-loop', 'bot-loop']

    
    file = h5py.File(directory_to_file+'/'+input_file_name+'.h5', 'r+')

    # Initialize dictionary to store all data
    strain = {}
    for segment in segments:
        strain[segment] = np.double(file[segment])


    time = file['time']
    # Convert decimated time data to datetime
    time_datetime = [datetime.datetime.fromtimestamp(i/1000000) for i in time]
    # convert h5 group to double
    return strain,time_datetime,time

def min_max(directory_to_file, input_file_name):
    ''' Function to get the minimum and maximum strain envelopes for plotting purposes
    Args:
        directory_to_file:
        input_file_name:

    Returns:
    '''
    file = h5py.File(directory_to_file+'/'+input_file_name+'.h5', 'r+')
    top_loop = file['top-loop']
    bot_loop = file['bot-loop']
    time = file['time']
    return


def min_max_luna_data(directory_to_file,input_file_name):
    ''' Function to get the minimum and maximum of preprocessed luna data
    Args:
        directory_to_file: full directory to file (string)
        input_file_name: .h5 file that you want to decimate (string)

    Returns:
        strain: dictionary of different segments containing numpy double of strain data that are now filtered
        time: list of datetimes

    Raises:
    '''
    # list of tower segments NOTE: Should eventually put this in a class or make it a global variable?
    segments = ['top-loop', 'bot-loop']
    
    # load in data
    strain, _,_ = load_preprocessed_luna_data(directory_to_file=directory_to_file,
                               input_file_name=input_file_name)
    
    # initialize min and max dictionaries
    min_strain = {}
    max_strain = {}

    for segment in segments:
        min_strain[segment] = np.nanmin(a = strain[segment],
                                        axis = 0)
        max_strain[segment] = np.nanmax(a = strain[segment],
                                        axis = 0)
    
    # Save the data arrays
    with h5py.File(directory_to_file + '/' + input_file_name + '_envelopes.h5', 'w') as hf:
        for segment in segments:
            hf.create_dataset(segment+'min_strain',data=min_strain[segment])
            hf.create_dataset(segment+'max_strain',data=max_strain[segment])
        hf.close()

    return

# Helper function to load min and max strain
def load_min_max_luna_data(directory_to_file,input_file_name):
    ''' Function to load min and max strain envelopes and process the datetimes
    Args:
        directory_to_file: full directory to file (string)
        input_file_name: .h5 file that you want to decimate (string)

    Returns:
        strain: dictionary of different segments containing numpy double of strain data
        time: list of datetimes

    Raises:
    '''
    # list of tower segments
    segments = ['top-loop', 'bot-loop']
    
    file = h5py.File(directory_to_file+'/'+input_file_name+'.h5', 'r+')

    # Initialize dictionary to store all data
    min_strain = {}
    max_strain = {}

    for segment in segments:
        min_strain[segment] = np.double(file[segment+'min_strain'])
        max_strain[segment] = np.double(file[segment+'max_strain'])

    # convert h5 group to double
    return min_strain,max_strain

