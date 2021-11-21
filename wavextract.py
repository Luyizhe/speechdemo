import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import os
np.set_printoptions(suppress=True)


# fig, ax = plt.subplots()
#
#
# wave_data, sr=librosa.load("E:\\NLP\SpeechEmotion\dataset\\angry\\13.wav",sr=None)
#
# wave_data1 = librosa.effects.preemphasis(wave_data, coef=-0.1)
# N=len(wave_data)
# Y1=np.fft.fft(wave_data1)
# Y2=np.fft.fft(wave_data)
# Fs=np.arange(np.floor(N/2))/N*sr
#
#
# melcof=librosa.feature.mfcc(wave_data,sr=sr,n_mfcc=40,n_fft=400, hop_length=160, win_length=400)
# print(sr)
# print(len(wave_data))
# print(np.shape(melcof))
path="E:\\NLP\SpeechEmotion\demo\IS09Feature\\"
features_total=[]
for i in os.listdir(path):
    file=open(path+i,'r')
    last_line = file.readlines()[-1]
    features = last_line.split(",")
    print(np.array(features[1:-1], dtype="float64"))
    exit()
    features_total.append(np.array(features[1:-1], dtype="float64"))
cmvn_mean=np.mean(features_total,axis=0)
cmvn_var=np.var(features_total,axis=0)
cmvn_std=np.std(features_total,axis=0)
print(cmvn_std)
print(cmvn_var)
out=(features_total-cmvn_mean)/(cmvn_var+1e-7)
print(np.max(out,axis=0))
