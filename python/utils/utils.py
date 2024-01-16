import h5py
import numpy as np
from scipy.signal import decimate
from scipy.signal import butter
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime
import os


def eda_raw_data(directory_to_file,name_of_file):
    ''' Function to open .h5 file
    Args:
        directory_to_file: full directory to file (string)
        name_of_file: .h5 file that you want to examine (string)
    Returns:
    
    Raises:
    
    '''
    file = h5py.File(directory_to_file+'/'+name_of_file+'.h5', 'r')
    # Read the dataset
    raw_data = file['/Acquisition/Raw[0]/RawData']
    raw_data = np.double(raw_data) # Convert to double


    return raw_data

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

    # UNWRAP PHASE_DECIM (RADIAN NUMBER) BEFORE DECIMATION 
    phase_data_unwrapped = np.unwrap(phase_data,axis=1) # axis on time dimension

    # Decimate
    # 100,000 Hz -> 100 Hz, decim_factor = 1000
    # Get length of decimated vector
    decim_length = len(decimate(phase_data_unwrapped[0,:], decim_factor))
    # Initialize phase_decim array
    # Use empty to make sure that we don't impute values
    phase_data_unwrapped_decim = np.empty((nch, decim_length))
    time_decim = np.empty((decim_length))

    # decim
    # Decimate each channel time series 
    for i in range(nch): # range(nch)
        phase_data_unwrapped_decim[i, :] = decimate(raw_data[i,:], decim_factor)

    # Decimate TIME data DON'T use decimate function! The decimate function downsamples the signal
    # after applying an anti-aliasing filter! Just take every decim_factor (1000th) entry
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
    #     Lgauge = 1.0209523
    Lgauge = 8.167619
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
    ''' Function to load decimated 100 Hz and process the datetimes
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
    time_datetime = [datetime.datetime.fromtimestamp(i/1000000) for i in time]
    # convert h5 group to double
    return np.double(data),time_datetime

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


def concatenate_and_save_h5(directory_to_file, output_filename, filetype):
    ''' Function to compile the decimated files
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



