#从各种pkl中提取需要的部分，以及更改需要替换的部分
import pandas as pd
import numpy as np
import pickle

videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = pickle.load(
    open(r"E:\NLP\SpeechEmotion\SpeechEmotionByself\IEMOCAP_features_BertText_4Class.pkl", "rb"), encoding='latin1')
#
#
videoIDsTmp={}
videoAudioTmp={}
videoTextTmp={}
videoLabelsTmp={}
trainVidtmp=[]
testVidtmp=[]
# file=open("IEMOCAP_vad_label.txt",'r')
# # file_clean=open("decoder_clean.txt",'w')
# v={}
# for lines in file.readlines():
#     sentence=lines.split("\t")[0]
#     vad=lines.split("\t")[1]
#     v[sentence]=vad.split(" ")[0]   #获得vad标签

for vid in videoIDs:
    videoIDsTmp[vid] = []
    videoLabelsTmp[vid] =[]
    videoTextTmp[vid] = []
    videoAudioTmp[vid] = []
    # print(videoIDs[vid])
    # print(videoLabels[vid])
    for clean in range(len(videoLabels[vid])):
        if videoLabels[vid][clean] != 5:
            if videoLabels[vid][clean] == 4:
                videoLabels[vid][clean] = 0
            videoIDsTmp[videoIDs[vid][clean]]=videoIDs[vid][clean]
            if vid in trainVid:
                trainVidtmp.append(videoIDs[vid][clean])
            elif vid in testVid:
                testVidtmp.append(videoIDs[vid][clean])
            videoLabelsTmp[videoIDs[vid][clean]]=videoLabels[vid][clean]
            AudioFeature=np.load(r'E:\NLP\SpeechEmotion\demo\\wav2vec2Feature\%s.npy'%(videoIDs[vid][clean]))
            videoAudioTmp[videoIDs[vid][clean]]=AudioFeature
            TextFeature=np.load(r'E:\NLP\SpeechEmotion\demo\BertEmbedding\%s.npy'%(videoIDs[vid][clean]))
            videoTextTmp[videoIDs[vid][clean]]=TextFeature
pickle.dump((videoIDsTmp,videoLabelsTmp,videoAudioTmp,videoTextTmp,trainVidtmp, testVidtmp),open(r"E:\NLP\SpeechEmotion\demo\\totalFeature.npy", "wb"),protocol=pickle.HIGHEST_PROTOCOL)
