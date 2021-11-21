import torch
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC,Wav2Vec2Config,Wav2Vec2Model
import soundfile as sf
import pickle
import csv


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from transformers import Wav2Vec2Processor, Wav2Vec2Model




processor = Wav2Vec2Processor.from_pretrained(r'E:\NLP\SpeechEmotion\demo\wav2vec2-base-960h')
model = Wav2Vec2Model.from_pretrained(r'E:\NLP\SpeechEmotion\demo\wav2vec2-base-960h')

root_path = "E:\\NLP\SpeechEmotion\dataset\IEMOCAP_full_release\\"
path = os.listdir(root_path)
#file_label = open("IEMOCAP_vad_label.txt", "w")
mfcc_total=[]
fbank_total=[]
for sessions in path:
    if not sessions.endswith(".pkl"):
        session_path = root_path + sessions + '\sentences\\wav\\'
        for script in os.listdir(session_path):
            wav_path=session_path+script+'\\'
            for wav in os.listdir(wav_path):
                try:
                    tmp, sr = sf.read(wav_path+wav)
                except:
                    print(wav_path+wav)
                inputs  = processor(tmp, sampling_rate=sr,return_tensors="pt")
                outputs = model(**inputs)
                # mfcc=librosa.feature.mfcc(tmp,sr=sr,n_mfcc=40, n_fft=len(tmp), hop_length=len(tmp)+1, win_length=None)
                #mfcc_total.append(mfcc)
                #print(wav)
                WavEmbedding=outputs.last_hidden_state.squeeze(0)
                np.save('fbank\\' + wav.split('.')[0] + '.npy', WavEmbedding.detach().numpy())
