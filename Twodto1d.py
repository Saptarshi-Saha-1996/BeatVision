from Onedto2d import BeatVision
import neurokit2 as nk
import numpy as np
from ecg_plot import plot_1
import matplotlib.pyplot as plt

filename = "A01443"
filepath = "/home/saptarshi/Research/CF_Explanation/ISI_2D_to_1D_data/" + str(filename) + ".npy"
#ecg = nk.ecg_simulate(duration=10, heart_rate=72,random_state=42)
#filepath   = "two_image.png"
ecg = np.load(filepath)
#print(len(ecg))



two_D_input, ZERO_input = BeatVision(ecg_signal=ecg, epsilon=20,delta=512)

def inv_BeatVision(two_D,ZERO,epsilon=10,delta=900):
    # Implement 2d to 1d 
    one_D = []
    for i,row in enumerate(two_D):
        nzeros = ZERO[i]    # remove zero paddings
        row = row[:-nzeros]
        if i == 0:      # paste the 1st segment
            one_D.extend(row[:-epsilon])   # using extend instead of append to make list

        else:
            one_D.extend(row[epsilon:-epsilon])
    return one_D

#print(inv_BeatVision(two_D=two_D_input,ZERO=ZERO_input,epsilon=10,delta=900 ))
back_to_1D = inv_BeatVision(two_D=two_D_input,ZERO=ZERO_input,epsilon=20,delta=512 )
#print(len(inv_BeatVision(two_D=two_D_input,ZERO=ZERO_input,epsilon=20,delta=512 )))

def plot_1D(ecg_1D,savepath):
# Plot 1D ecg
    ecg_signal = ecg_1D
    rsignals, rpeaks = nk.ecg_peaks(ecg_signal, method='neurokit',sampling_rate=300)
    signal_dwt, waves_dwt = nk.ecg_delineate(ecg_signal, 
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
    fig, ax = plt.subplots(1,1,figsize=(25, 2.5))
    plot_1(ecg=ecg_1D, attention =None, attention_area=None, top_k= None,
            peaks= peaks,valid_p_peaks=None,RR_length_outlier=None,peaks_index=index,at_thres=0.0,
            fig=fig, ax=ax, 
            sample_rate=300,line_w=1.0,
            title = 'ECG',
            ecg_amp=1.2,timetick=0.2)
    plt.savefig(savepath)

plot_1D(ecg,savepath="original_1D.jpg")
plot_1D(back_to_1D,savepath="back_to_1D.jpg")
print(len(ecg))
print(len(back_to_1D))


