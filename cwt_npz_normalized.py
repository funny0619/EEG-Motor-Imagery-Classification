import scipy.io as sio
import numpy as np
import os
import pandas
import matplotlib.pyplot as plt
from scipy import signal
import pywt
mat_dict = sio.loadmat("data.mat")
data = mat_dict["data"]
time_series = mat_dict["time_points"]
labels = mat_dict["label"]
channel_labels = mat_dict['channel_labels']
data = np.swapaxes(data.T,1,2)

def cwt_wavelet_transform(data):
    num = len(data)

    wavelet = 'morl' # wavelet type: morlet
    fs = 250.0 # sampling frequency: 8KHz # scales for morlet wavelet 
    t,dt = np.linspace(-0.2,3,801,retstep=True)
    w = 5
    scalogram_data = np.zeros([num,24,5,801])
    # freq = np.array([1,2,4,7,12,18,25,32,40,45])
    freq = np.linspace(1,45,45)

    widths = w*fs / (2*freq*np.pi)
    print("These are the scales that we are using: ", widths)

    frequencies = pywt.scale2frequency(wavelet, widths) / dt # Get frequencies corresponding to scales
    print("These are the frequencies that re associated with the scales: ", frequencies)
    

    adelta = np.zeros([801])
    atheta = np.zeros([801])
    aalpha = np.zeros([801])
    abeta = np.zeros([801])
    agamma = np.zeros([801])
    
    for i in range(num):
        if i % 100 == 0:
            print("Now on " + str(i))
        for j in range(24):
            sig = data[i,j]
            wavelet_coeffs, freqs = pywt.cwt(sig, widths, wavelet = wavelet, sampling_period=dt)
            
            wavelet_coeffs = np.abs(wavelet_coeffs)


            for k in range(801):
                
                adelta[k] = np.mean(wavelet_coeffs[:4,k])
                atheta[k] = np.mean(wavelet_coeffs[4:8,k])
                aalpha[k] = np.mean(wavelet_coeffs[8:14,k])
                abeta[k] = np.mean(wavelet_coeffs[14:30,k])
                agamma[k] = np.mean(wavelet_coeffs[30:45,k])
            
            delta = (adelta-np.min(adelta))/(np.max(adelta)-np.min(adelta))
            theta = (atheta-np.min(atheta))/(np.max(atheta)-np.min(atheta))
            alpha = (aalpha-np.min(aalpha))/(np.max(aalpha)-np.min(aalpha))
            beta = (abeta-np.min(abeta))/(np.max(abeta)-np.min(abeta))
            gamma = (agamma-np.min(agamma))/(np.max(agamma)-np.min(agamma))

            for l in range(5):
                if l == 0:
                    scalogram_data[i,j,l] = delta
                if l == 1:
                    scalogram_data[i,j,l] = theta
                if l == 2:
                    scalogram_data[i,j,l] = alpha
                if l == 3:
                    scalogram_data[i,j,l] = beta
                if l == 4:
                    scalogram_data[i,j,l] = gamma
    return scalogram_data

transformed_data = cwt_wavelet_transform(data)
print(transformed_data.shape)
print(transformed_data)

name_dict = {
    'data' : transformed_data,
    'labels' : labels,
}

np.savez("cwt_normalized_data",**name_dict)

