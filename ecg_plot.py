import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os
from math import ceil 
from matplotlib.colors import Normalize
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle



def _ax_plot(ax, x, y, secs=10, lwidth=0.7, amplitude_ecg = 1, time_ticks =0.2,grid = True):
    ax.set_xticks(np.arange(0,secs+1,time_ticks))  # done secs + 1   
    ax.set_yticks(np.arange(-ceil(amplitude_ecg),ceil(amplitude_ecg),1.0))

    #ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax.minorticks_on()
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(5)) #auto minor locator was 7

    ax.set_ylim(-amplitude_ecg, amplitude_ecg)
    ax.set_xlim(0, secs)

    
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))

    ax.plot(x,y, linewidth=lwidth)


def attention_plot(ax,x,y,threshold):  #If threshold != 0 no colour gradient
    if threshold != 0:                # Threshold = 0 to see original attentions 
        y = y > threshold
    
    ax = plt.plot(x,np.array(y)-1,c='darkred',linewidth=2)
    #plt.scatter(x,y,c=y cmap='Greens', edgecolors='k',  marker='|', s=10000,vmin=0,vmax=1)
    #plt.fill_between(x=x,y1=y)
    #plt.scatter(x,-np.array(y),c=y,marker='.',s=1,cmap='Greens')
    #
    #cbar = plt.colorbar(label='Color Gradient (0 to 1)')

def plot_peaks(ax,peaks):
    for x in peaks:
        ax.vlines(x, ymin=-1, ymax=1, colors='black', linestyles='dotted')

def plot_1( ecg,
            at_thres,
            fig, 
            ax ,
            attention, 
            attention_area=None, 
            top_k=None, 
            peaks=None,
            peaks_index=None,
            valid_p_peaks =None,   #mark r peaks with invalid p peaks
            RR_length_outlier =None,  # mark abnormal RR interval lenghts
            sample_rate=300,
            title = 'ECG', 
            fig_width = 15, 
            fig_height = 5, 
            line_w = 1,
            ecg_amp = None, # changed from 1.8 to 1
            timetick = 0.2):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        attention   : Attention array
        peaks      : Peaks as an array
        at_thes     : The threshold for attention to plot (attention =1 for att > at_thres)
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        fig_width  : The width of the plot
        fig_height : The height of the plot
        valid_p_peaks  : Bool array having 1 when p peak is valid
        RR_lenght_outlier   : Bool array having 1 when RR interval length is abnormal
    """
    #plt.figure(figsize=(fig_width,fig_height))
    #plt.suptitle(title)
    #plt.subplots_adjust(
    #    hspace = 0, 
    #    wspace = 0.04,
    #    left   = 0.04,  # the left side of the subplots of the figure
    #    right  = 0.98,  # the right side of the subplots of the figure
    #    bottom = 0.2,   # the bottom of the subplots of the figure
    #    top    = 0.88
    #    )
    seconds = len(ecg)/sample_rate
    ecg_amp = np.max(ecg) + 0.2        # setting amplitude
    #ax = plt.subplot(1, 1, 1)
    #plt.rcParams['lines.linewidth'] = 5
    step = 1.0/sample_rate
    _ax_plot(ax,np.arange(0,len(ecg)*step,step),ecg, seconds, line_w, ecg_amp,timetick)
    # if attention is not None:
    #     _ax_plot(ax,np.arange(0,len(attention)*step,step), -attention,seconds,line_w, ecg_amp,timetick)
    #     if attention_area is not None:
    #         attention_plot(ax,np.arange(0,len(attention_area)*step,step),attention_area,threshold = 0.0) #-1*attention*(attention>at_thres)
            
    #         # commented out for removing attention curve
    #         custom_cmap = LinearSegmentedColormap.from_list('cmap',['white','green'])
           
    #         normalize_ = mpl.colors.Normalize(vmin=np.array(attention_area).min(), vmax=np.array(attention_area).max())
    #         x_=np.arange(0,len(attention)*step,step)
    #         for i in range(len(attention_area)-1):
    #             plt.fill_between(x=[x_[i],x_[i+1]],y1=[-attention[i],-attention[i+1]],  color=custom_cmap(normalize_(attention_area[i])))
            
    #         cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=normalize_))
    #         cbar.set_label('Color Gradient (0 to 1)')

    if peaks is not None:
        
        colors= ['purple','darkorange','green','indianred','black']
        j=0
        #for peak in peaks.keys():
            #ax.plot(np.arange(0,len(ecg)*step,step),peaks[peak]*ecg,color='black', linewidth=0.75,linestyle='solid')
            #if peaks_index is not None:
                #for index in peaks_index[peak]:
                    
                    #if peak == 'R':
                        #ys = peaks[peak]*ecg
                        #ax.text(index/sample_rate,ys[index],s=peak,size=5.0,ha='left',)
                        
                    #elif peak == 'T':
                        #ys = peaks[peak]*ecg
                        #ax.text(index/sample_rate,ys[index],s=peak,size=5.0,ha='left',)
                    #elif peak == 'Q':
                        #ys = peaks[peak]*ecg
                        #ax.text(index/sample_rate,ys[index]+0.07,s=peak,size=5.0,ha='center',)

                    #else:
                        #ax.text(index/sample_rate,-0.1,s=peak,size=5.0,ha='center',)
            #j=j+1
        eps=0.05
        if top_k is not None:
            left_r_peak=(np.array(peaks_index['R'])[top_k]/sample_rate) -eps
            right_r_peak = np.array(peaks_index['R'])[top_k+1]/sample_rate
            width = right_r_peak-left_r_peak + eps
            #print(len(right_r_peak),left_r_peak)
            for i in range(len(left_r_peak)):
                ax.add_patch(Rectangle( (left_r_peak[i],0),height=1.1,width=width[i],alpha=0.4,color='red'))
            #print(np.array(top_k)/sample_rate,(np.array(top_k)+1)/sample_rate)

        # Marking R peaks with no p peak just before
        if valid_p_peaks is not None:
            r_peak_index = []
            r_peak_value = []
            for j in range(len(valid_p_peaks)):
                if valid_p_peaks[j] == 0:   # marking R peaks with invalid p-peaks
                    r_peak_index.append(peaks_index['R'][j]/sample_rate)
                    r_peak_value.append(ecg[peaks_index['R'][j]])
            #for x in r_peak_index:
                #print(x)
            ax.scatter(r_peak_index,r_peak_value,color='green',s=100, alpha=0.5)

        if RR_length_outlier is not None:
            for j in range(len(RR_length_outlier)-1):
                if RR_length_outlier[j]==1:
                    ax.hlines(y = -1,xmin=peaks_index['R'][j]/sample_rate,xmax=peaks_index['R'][j+1]/sample_rate,color='blue',alpha=0.8,linewidth=3)



    return ax