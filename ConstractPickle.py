import os
import numpy as np
import pickle
import random
import csv


segment={"-1.0":0,"-0.8":0,"-0.6":1,"-0.4":1,"-0.2":1,"0.0":2,"0.2":3,"0.4":3,"0.6":3,"0.8":4,"1.0":4}
trainVidtmp=[]
testVidtmp=[]
videoIDsTmp = {}
videoSpeakersTmp = {}
videoLabelsTmp = {}
videoTextTmp = {}
videoAudioTmp = {}
videoSentenceTmp = {}
csvfile=csv.reader(open(r"E:\NLP\SpeechEmotion\dataset\SIMS\SIMS-label.csv",'r', encoding="utf-8"))
flag=0
for vid in csvfile:
    if flag==1:
        ID=vid[0]+"_"+vid[1]
        videoIDsTmp[ID]=ID
        if vid[3]=="train" or vid[8]=="valid":
            trainVidtmp.append(ID)
        else:
            testVidtmp.append(ID)
        videoLabelsTmp[ID]=segment[vid[3]]
        TextFeature = np.load(r'E:\NLP\SpeechEmotion\demo\BertEmbedding\%s.npy' % (ID))
        videoTextTmp[ID] = TextFeature
        AudioFeature = np.load(r'E:\NLP\SpeechEmotion\demo\fbank\%s.npy' % (ID))
        videoAudioTmp[ID] = AudioFeature
    flag=1
pickle.dump((videoIDsTmp, videoLabelsTmp, videoAudioTmp, videoTextTmp, trainVidtmp, testVidtmp),
                open(r"E:\NLP\SpeechEmotion\demo\\totalFeature.npy", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    # videoIDsTmp[vid] = []
    # videoSpeakersTmp[vid] = []
    # videoLabelsTmp[vid] = []
    # videoTextTmp[vid] = []
    # videoAudioTmp[vid] = []
    # videoSentenceTmp[vid] = []

#
# videoIDsTmp = {}
# videoSpeakersTmp = {}
# videoLabelsTmp = {}
# videoTextTmp = {}
# videoAudioTmp = {}
# videoSentenceTmp = {}
# trainVidTmp = []
# testVidTmp = []
#
# for vid in videoIDs:
#     videoIDsTmp[vid] = []
#     videoSpeakersTmp[vid] = []
#     videoLabelsTmp[vid] = []
#     videoTextTmp[vid] = []
#     videoAudioTmp[vid] = []
#     videoSentenceTmp[vid] = []
#
#     for clean in range(len(videoLabels[vid])):
#
#         if videoLabels[vid][clean] != 5:
#             if videoLabels[vid][clean] == 4:
#                 videoLabels[vid][clean] = 0
#             videoIDsTmp[vid].append(videoIDs[vid][clean])
#             videoSpeakersTmp[vid].append(videoSpeakers[vid][clean])
#             videoLabelsTmp[vid].append(videoLabels[vid][clean] )
#             # videoLabelsTmp[vid].append(int(v[videoIDs[vid][clean]])-1)
#             videoTextTmp[vid].append(videoText[vid][clean])
#             path="E:\\NLP\SpeechEmotion\demo\\fbank\\"
#             fbank=np.load(path+videoIDs[vid][clean]+'.npy').squeeze(0)
#             #videoAudioTmp[vid].append(videoAudio[vid][clean])
#             videoAudioTmp[vid].append(fbank.astype(np.float32))
#             videoSentenceTmp[vid].append(videoSentence[vid][clean])
# pickle.dump(
#     (videoIDsTmp, videoSpeakersTmp, videoLabelsTmp, videoTextTmp, videoAudioTmp,videoVisual, videoSentenceTmp, trainVid, testVid),
#     open("./IEMOCAP_features_BertText_4Class_fbank.pkl", "wb"),protocol=pickle.HIGHEST_PROTOCOL)
