import torch
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pickle
import csv


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(r"E:\NLP\SpeechEmotion\demo\chinese-roberta-wwm-ext")
model = AutoModel.from_pretrained(r"E:\NLP\SpeechEmotion\demo\chinese-roberta-wwm-ext")

videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = pickle.load(
    open("E:\\NLP\SpeechEmotion\SpeechEmotionByself\IEMOCAP_features_raw.pkl", "rb"), encoding='latin1')
#
#


for vid in videoIDs:
    for clean in range(len(videoLabels[vid])):
        if videoLabels[vid][clean] != 5:
            if videoLabels[vid][clean] == 4:
                videoLabels[vid][clean] = 0
            DicID = tokenizer(videoSentence[vid][clean], return_tensors="pt")
            TextEmbedding = model(DicID['input_ids'])
            np.save('BertEmbedding\\' + videoIDs[vid][clean]+ '.npy', TextEmbedding[0].detach().numpy())
# root_path = r"E:\NLP\SpeechEmotion\dataset\SIMS\Raw\audio\\"
# path=os.listdir(root_path)
# csvfile=csv.reader(open(r"E:\NLP\SpeechEmotion\dataset\SIMS\SIMS-label.csv",'r', encoding="utf-8"))
# for tmp in csvfile:
#     DicID = tokenizer(tmp[2], return_tensors="pt")
#     TextEmbedding = model(DicID['input_ids'])
#     np.save('BertEmbedding\\' + tmp[0]+"_"+tmp[1] + '.npy', TextEmbedding[0].detach().numpy())
#





