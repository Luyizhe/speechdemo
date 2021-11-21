#从各种pkl中提取需要的部分，以及更改需要替换的部分
import pandas as pd
import numpy as np
import pickle

videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = pickle.load(
    open("./IEMOCAP_features_BertText_4Class.pkl", "rb"), encoding='latin1')
#
#
videoIDsTmp = {}
videoSpeakersTmp = {}
videoLabelsTmp = {}
videoTextTmp = {}
videoAudioTmp = {}
videoSentenceTmp = {}
trainVidTmp = []
testVidTmp = []

for vid in videoIDs:
    videoIDsTmp[vid] = []
    videoSpeakersTmp[vid] = []
    videoLabelsTmp[vid] = []
    videoTextTmp[vid] = []
    videoAudioTmp[vid] = []
    videoSentenceTmp[vid] = []

    for clean in range(len(videoLabels[vid])):

        if videoLabels[vid][clean] != 5:
            if videoLabels[vid][clean] == 4:
                videoLabels[vid][clean] = 0
            videoIDsTmp[vid].append(videoIDs[vid][clean])
            videoSpeakersTmp[vid].append(videoSpeakers[vid][clean])
            videoLabelsTmp[vid].append(videoLabels[vid][clean] )
            # videoLabelsTmp[vid].append(int(v[videoIDs[vid][clean]])-1)
            videoTextTmp[vid].append(videoText[vid][clean])
            path="E:\\NLP\SpeechEmotion\demo\\wav2vec2Feature\\"
            fbank=np.load(path+videoIDs[vid][clean]+'.npy').squeeze(0)
            #videoAudioTmp[vid].append(videoAudio[vid][clean])
            videoAudioTmp[vid].append(fbank.astype(np.float32))
            videoSentenceTmp[vid].append(videoSentence[vid][clean])
pickle.dump(
    (videoIDsTmp, videoSpeakersTmp, videoLabelsTmp, videoTextTmp, videoAudioTmp,videoVisual, videoSentenceTmp, trainVid, testVid),
    open("./IEMOCAP_features_BertText_4Class_wav2vec2.pkl", "wb"),protocol=pickle.HIGHEST_PROTOCOL)

