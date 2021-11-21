import os
import librosa
import numpy as np
import python_speech_features
np.set_printoptions(suppress=True)

######################################提取CASIA音频特征########################################
root_path = r"E:\NLP\SpeechEmotion\dataset\SIMS\Raw\audio\\"
path=os.listdir(root_path)
for wav in path:
    SpeakerPath=root_path+wav
    tmp, sr = librosa.load(SpeakerPath, sr=None)
    y_16k = librosa.resample(tmp, sr, 16000)
    fbank,_=python_speech_features.base.fbank(y_16k, samplerate=16000, winlen=0.1, winstep=0.05, nfilt=80, nfft=2048, lowfreq=0, highfreq=8000, preemph=0.97, winfunc=np.hamming)
    np.save('fbank\\' + wav.split('.')[0] + '.npy', fbank)


####################################提取IEMOCAP音频特征#############################################################
# root_path = "E:\\NLP\SpeechEmotion\dataset\IEMOCAP_full_release\\"
# path = os.listdir(root_path)
# #file_label = open("IEMOCAP_vad_label.txt", "w")
# mfcc_total=[]
# fbank_total=[]
# for sessions in path:
#     session_path = root_path + sessions + '\sentences\\wav\\'
#     for script in os.listdir(session_path):
#         wav_path=session_path+script+'\\'
#         for wav in os.listdir(wav_path):
#             try:
#                 tmp,sr=librosa.load(wav_path+wav,sr=None)
#             except:
#                 print(wav_path+wav)

#             fbank,_=python_speech_features.base.fbank(tmp, samplerate=sr, winlen=0.1, winstep=0.05, nfilt=80, nfft=2048, lowfreq=0, highfreq=8000, preemph=0.97, winfunc=np.hamming)
#             # mfcc=librosa.feature.mfcc(tmp,sr=sr,n_mfcc=40, n_fft=len(tmp), hop_length=len(tmp)+1, win_length=None)
#             #mfcc_total.append(mfcc)
#             #print(wav)
#             np.save('fbank\\' + wav.split('.')[0] + '.npy', fbank)
