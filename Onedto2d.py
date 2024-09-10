#bring the plotter

import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
# Take an ECG signal 

#Ecg_signal_path of 1d 
filename = "A00064"
filepath = "/home/saptarshi/Research/CF_Explanation/ISI_2D_to_1D_data/" + str(filename) + ".npy"
#ecg = nk.ecg_simulate(duration=10, heart_rate=72,random_state=42)
ecg = np.load(filepath)
#print(len(ecg))

#signal path 
#ecg = nk.ecg_simulate(duration=10, heart_rate=72,random_state=42)
#print(ecg)
def BeatVision(ecg_signal,epsilon=10,delta=None):
    # R peak detection 
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=300, method="hamilton2002")
    rpeaks = rpeaks['ECG_R_Peaks']
    segments =[]
    #print(len(rpeaks))
    for j,peak in enumerate(rpeaks):  
        #  if j==0:
        #     segments.append(ecg_signal[: rpeaks[j+1]+epsilon])
        #  elif j== len(rpeaks)-2:
        #     segments.append(ecg_signal[rpeaks[j]-epsilon:] )
        if j< len(rpeaks)-1 :
            segments.append(ecg_signal[rpeaks[j]-epsilon:rpeaks[j+1]+epsilon])
            #if j == 0:
                #head = ecg_signal[:rpeaks[j]-epsilon]
            #if j == len(rpeaks)-2:
                #tail = ecg_signal[rpeaks[j+1]+epsilon:]
    print(len(segments),len(rpeaks)-1)
    assert len(segments) == len(rpeaks)-1
    
    max_length = max(segment.shape[0] for segment in segments)
    print(max_length)
    assert delta >= max_length
    #print(max_length)

    ## Zero padding by delta
    ZERO = []
    for j,segment in enumerate(segments):
        if delta > len(segment):
            n_zero =delta - len(segment)
            #print(n_zero)
            ZERO.append(n_zero)
            segments[j] = np.append(segment,np.zeros(n_zero)).reshape(1,-1)
    assert len(ZERO) == len(segments)
    # for segment in segments:
    #     print(segment.shape)
    two_D = np.vstack(segments)

    return two_D, ZERO , max_length , len(segments)  #head, tail

# find reasonable epsilon and delta
#print(BeatVision(ecg_signal=ecg, epsilon=10,delta=900))
#atwo_D,ZERO = BeatVision(ecg_signal=ecg, epsilon=20,delta=512)

def plot_2D(matrix):
    plt.imshow(matrix, cmap='seismic', interpolation='nearest',aspect=8)
    #plt.colorbar()
    plt.savefig("two_image.pdf")
    plt.show()


# Plot the matrix
#plot_2D(two_D)
