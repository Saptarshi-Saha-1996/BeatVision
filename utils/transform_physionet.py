folderpath = '/home/saptarshi/Research/encoder_decoder/physionet2017challenge/training2017/'
labelpath ='/home/saptarshi/Research/encoder_decoder/physionet2017challenge/REFERENCE-original.csv'
import sys
sys.path.append('/home/saptarshi/Research/encoder_decoder/')
import pandas as pd
import csv
import yaml

EPSILON = 50
DELTA = 768

labels = pd.read_csv(labelpath,header=None)
labels.columns = ['Name', 'Label']

label=labels.loc[labels['Name'] == "A00003", 'Label']
label.squeeze()
from neurokit2 import signal_filter

f_6000 =  [ 'A08112', 'A04652', 'A00420', 'A07728', 'A07893', 'A06787', 'A02514', 'A07995', 'A03719', 'A02408', 
            'A02738', 'A02821', 'A02364', 'A00307', 'A03572', 'A06434', 'A01427', 'A02552', 'A08221', 'A05581',]

scrutiny =   [  'A02706', 'A05116', 'A06435', 'A08092', 'A07136', 'A01965', 'A07978',
                'A04362', 'A05900', 'A07724', 'A01650', 'A01521', 'A05189', 'A08335',
                'A04452', 'A01134', 'A01329', 'A05956', 'A04756', 'A02784', 'A07320', 
                'A02955', 'A04216', 'A05170', 'A07890', 'A06103', 'A03870', 'A03806', 
                'A07518', 'A08521']
noises  = ['A07136', 'A08335', 'A05956', 'A04756', 'A04216', 'A06103'] # it doesn't mean that they belong to noise class

exclude = ['A05956', 'A04756', 'A04216']

import neurokit2 as nk
import scipy.io
import os
import numpy as np
from Onedto2d import BeatVision
l=0
MAX_Length = []
len_=100000
name_={}
for file in os.listdir(folderpath):
    if file.endswith('.mat'):
        name = os.path.splitext(file)[0]
        print(name)
        filepath = os.path.join(folderpath, file)
        data =  scipy.io.loadmat(filepath)
        data= np.array(data['val'],dtype=np.float32).flatten()
        data_nan = np.isnan(data)
        data[data_nan]=0
        len_ = data.shape[0]
        
        if name not in exclude:
            if name not in scrutiny :
                data = data[len_//10:(9*len_)//10]
                if len(data)>= 6000:
                    if name in f_6000:
                        data= data[0:6000]
                    else:
                        data= data[-6000:]
            if name in scrutiny:
                if name not in noises:
                    data = data[-3000:]  #data[len_//8 : 1+(7*len_)//8 ]
                else:
                    data = data[0:3000]

            data = nk.ecg_clean(data,sampling_rate=300,method ='hamilton2002',)
            normalized_data = (data-np.min(data))/(np.max(data)-np.min(data))
            normalized_data_nan = np.isnan(normalized_data)
            normalized_data[normalized_data_nan]=0
            np.save('/home/saptarshi/Research/encoder_decoder/physionet2017challenge/OneD_data/'+name, np.array(normalized_data))
            twoD, zero, max_length, no_seg = BeatVision(normalized_data, epsilon = EPSILON, delta = DELTA )
            np.savetxt('/home/saptarshi/Research/encoder_decoder/physionet2017challenge/training_CSV/'+name+'.csv',twoD,fmt='%.5f' )
            save_zero ='/home/saptarshi/Research/encoder_decoder/physionet2017challenge/Zeros/'+name+'.csv'
            with open(save_zero, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(zero)

            l = max(l,max_length)
            if max_length > 768:
                name_[name] = data.shape[0]
                    #labels.loc[labels['Name'] == name, 'Label'].squeeze()

            #print(name,max_length,l,no_seg)

print(name_,len(name_))
print(list(name_.keys()))








