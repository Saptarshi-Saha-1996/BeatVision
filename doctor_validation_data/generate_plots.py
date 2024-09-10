# Plot One 1D array of ECG with attention
import neurokit2 as nk
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 1500
import numpy as np
import matplotlib.pyplot as plt
from ecg_plot import plot_1

def plot_numbered(filename):
# Load the array from the .npy file
    file_path = '/home/saptarshi/Research/encoder_decoder/physionet2017challenge/OneD_data/' + str(filename) + '.npy'
    data = np.load(file_path)

    rsignals, rpeaks = nk.ecg_peaks(data, method='hamilton2002',sampling_rate=300)

# Visualize P-peaks and T-peaks
    signal_dwt, waves_dwt = nk.ecg_delineate(data, 
                                         rpeaks, 
                                         sampling_rate=300, 
                                         show_type='peaks',
                                         show=False)

    index = {"S":[x for x in waves_dwt['ECG_S_Peaks'] if x == x],
         "P":[x for x in waves_dwt['ECG_P_Peaks'] if x == x],
         "Q":[x for x in waves_dwt['ECG_Q_Peaks'] if x == x],
         "T":[x for x in waves_dwt['ECG_T_Peaks'] if x == x],
         "R": rpeaks['ECG_R_Peaks']
        }
    
    peaks ={"S":np.array(signal_dwt["ECG_S_Peaks"]),
        "P":np.array(signal_dwt["ECG_P_Peaks"]),
        "Q":np.array(signal_dwt["ECG_Q_Peaks"]),
        "T":np.array(signal_dwt["ECG_T_Peaks"]),
        "R":np.array(rsignals["ECG_R_Peaks"])
        }

    fig, ax = plt.subplots(1,1,figsize=(10, 2))
    plot_1(ecg=data, attention =None, attention_area=None, top_k= None,
            peaks= peaks,valid_p_peaks=None,RR_length_outlier=None,peaks_index=index,at_thres=0.0,
            fig=fig, ax=ax, 
            sample_rate=300,line_w=1.0,
            title = 'ECG',
            ecg_amp=1.2,timetick=0.2,fig_height=2,fig_width=12,numbering=True)
    save_path = '/home/saptarshi/Research/encoder_decoder/doctor_validation_data/AF/' + str(filename)  + '.jpg'   #attention_area =mod_attention, attention =data2,top_k = top_k_indices
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

import pandas as pd
import os

# Load the CSV file
csv_file_path = '/home/saptarshi/Research/encoder_decoder/physionet2017challenge/REFERENCE-original.csv' 
data = pd.read_csv(csv_file_path,header=None)

filtered_data = data[data[1] == 'A']

search_directory = '/home/saptarshi/Research/encoder_decoder/physionet2017challenge/OneD_data'  

# Get the list of filenames from the filtered data
filenames_to_search = filtered_data[0].tolist()

# Search for the files in the specified directory and perform a task if found
for filename in filenames_to_search:
    file_path = os.path.join(search_directory, filename +'.npy')
    if os.path.exists(file_path):
        try:
        # Perform a certain task on the found file
            plot_numbered(filename)
        except ValueError as ve:
            print(f"ValueError encountered while processing {file_path}: {ve}")

    else:
        print(f"File not found: {filename}")
