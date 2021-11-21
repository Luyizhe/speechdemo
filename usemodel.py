import torch
import speech_recognition as sr
import argparse
from transformers import AutoTokenizer, AutoModel
import numpy as np
import python_speech_features
import os 
import librosa
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_model = torch.load('E:\\NLP\SpeechEmotion\\demo\\best.pt',map_location=torch.device(device))
map_location=torch.device('cpu')
test_model.eval()
# tokenizer = AutoTokenizer.from_pretrained(r"E:\\NLP\SpeechEmotion\\PretrainedModel\\uncased_L-12_H-768_A-12\\")
tokenizer = AutoTokenizer.from_pretrained(r"E:\NLP\SpeechEmotion\demo\chinese-roberta-wwm-ext")
# model = AutoModel.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained(r"E:\NLP\SpeechEmotion\demo\chinese-roberta-wwm-ext")

labels_order = ['hap', 'sad', 'neu', 'ang']

def GetTextEmbedding(wav):
    ASR=sr.Recognizer()
    audio=sr.AudioFile(wav)
    with audio as source :
        audiowav=ASR.record(source)
    text=ASR.recognize_google(audiowav, language='zh-cn')
    DicID=tokenizer(text, return_tensors="pt")
    Embedding=model(DicID['input_ids'])
    TextEmbedding=Embedding[0].detach().numpy()
    return TextEmbedding
  
def GetAudioEmbedding(wav,sample_rate):
    fbank,_=python_speech_features.base.fbank(wav, samplerate=sample_rate, winlen=0.1, winstep=0.05, nfilt=80, nfft=2048, lowfreq=0, highfreq=8000, preemph=0.97, winfunc=np.hamming)
    return fbank
    
def main(wavpath):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--wav', type=str, default=r"E:\\NLP\SpeechEmotion\dataset\\IEMOCAP_full_release\Session2\sentences\wav\Ses02F_impro01\Ses02F_impro01_F002.wav", help='input wav file')
    # parser.add_argument('--modal', type=str, default="tmp.pt", help='model file')
    # args = parser.parse_args()

    tmp,sample_rate=librosa.load(wavpath,sr=None)
    ################获得文本embedding######################
    TextEmbedding=GetTextEmbedding(wavpath)
    #################获得音频embedding########################### 
    AudioEmbedding=GetAudioEmbedding(tmp,sample_rate)
    #################获得模型所有输入##############################
    text=torch.from_numpy(TextEmbedding).squeeze(0)
    audio=torch.from_numpy(AudioEmbedding)
    seqlen_text=torch.tensor([len(text)])
    seqlen_audio=torch.tensor([len(audio)])
    text_mask=torch.ones(len(text))
    audio_mask=torch.ones(len(audio))
    ##################补足batch维度##################
    text_mask=text_mask[None,:]
    audio_mask=audio_mask[None,:]
    text=text[None,:,:]
    audio=audio[None,:,:]
    ###################运行模型#################################
    text=text.to(device)
    audio=audio.to(device)
    text_mask=text_mask.to(device)
    audio_mask=audio_mask.to(device)
    predict=test_model(audio.to(torch.float), text.to(torch.float), seqlen_audio, seqlen_text, audio_mask.to(torch.float),text_mask.to(torch.float))
    emotion=torch.argmax(predict)
    return labels_order[emotion.item()]
    
if __name__ == "__main__":
    predict=main(r"E:\NLP\SpeechEmotion\dataset\IEMOCAP_full_release\Session1\sentences\wav\Ses01F_impro01\Ses01F_impro01_F006.wav")
    print(predict)