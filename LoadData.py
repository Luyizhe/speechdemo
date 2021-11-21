import torch

import os
import logging
import numpy as np
import re
from torch.utils.data import Dataset
import copy
import pickle
from torch.nn import functional
import io
label_classes = 5
class Loadtest(Dataset):
    def __init__(self):
        self.len = 1

    def __getitem__(self, item):
        return torch.ones(3, 3)


class LoadDiaData(Dataset):
    def __init__(self, train_or_test,dataset="ground_truth",classify='emotion'):
        if classify=="emotion":
            if dataset=="speech_recognition":
                file = open('IEMOCAP_features_BertText4_ASR.pkl', 'rb')
            elif dataset=="google_cloud":
                file = open('IEMOCAP_features_BertText4_ASR_Google.pkl', 'rb')
            elif dataset=="ground_truth":
                #file = open('totalFeature.npy', 'rb')
                file = open('totalFeature.npy', 'rb')
            elif dataset=="resources":
                file = open("IEMOCAP_features_4Cate.pkl", 'rb')
        self.filename=file.name
        self.videoIDs,self.videoLabels, self.videoAudio, self.videoText,self.trainVid, self.testVid = pickle.load(file, encoding='latin1')
        self.indexes = np.arange(len(self.videoIDs))
        self.trainVid = list(self.trainVid)
        self.testVid = list(self.testVid)
        self.text_max = 0
        self.audio_max=0
        self.train_or_test = train_or_test
        for vid in self.trainVid + self.testVid:
            self.videoText[vid]=self.videoText[vid].squeeze(0)
            if len(self.videoText[vid]) > self.text_max:
                self.text_max = len(self.videoText[vid])
        for vid in self.trainVid + self.testVid:
            if len(self.videoAudio[vid]) > self.audio_max:
                self.audio_max = len(self.videoAudio[vid])


    def __getitem__(self, batch_index):

        indexes = self.indexes[batch_index]
        # 处理返回各种特征值
        if self.train_or_test == 'train':
            vid = self.trainVid[indexes]
        if self.train_or_test == 'test':
            vid = self.testVid[indexes]
        tmp = np.array(self.videoAudio[vid]).reshape(
            [np.shape(self.videoAudio[vid])[0], np.shape(self.videoAudio[vid])[1], 1])
        # 将音频特征处理为统一长度方便放入batch。
        audio_len = len(self.videoAudio[vid])
        gap = self.audio_max - audio_len
        #print(gap)
        audio_feat = np.pad(tmp, [(0, gap), (0, 0), (0, 0)], mode='constant')
        audio = [torch.tensor(audio_feat[:, :, 0]), torch.tensor(audio_len)]

        # 将文本特征处理为统一长度方便放入batch。

        # if self.filename=="IEMOCAP_features_BertText4_ASR.pkl":
        #     print(np.array(self.videoText[vid]).shape)
        self.videoText[vid]=np.array(self.videoText[vid])
        if len(np.shape(self.videoText[vid]))!=2:
            self.videoText[vid]=self.videoText[vid].squeeze(0)

        tmp = np.array(self.videoText[vid]).reshape(
            [np.shape(self.videoText[vid])[0], np.shape(self.videoText[vid])[1], 1])
        text_len = len(self.videoText[vid])
        gap = self.text_max - text_len

        text_feat = np.pad(tmp, [(0, gap), (0, 0), (0, 0)], mode='constant')
        text = [torch.tensor(text_feat[:, :, 0]), torch.tensor(text_len)]
        # 将label处理为统一长度方便放入batch。
        # tmp = np.array(self.videoLabels[vid]).reshape(
        #     [np.shape(self.videoLabels[vid])[0], 1])
        # labels = np.pad(tmp, [(0, gap), (0, 0)], mode='constant', constant_values=(3, 3))
        # labels = torch.LongTensor(labels)
        label=self.videoLabels[vid]
        audio_mask = np.zeros(np.shape(audio[0])[0])
        audio_mask[:audio[1]] = 1
        text_mask = np.zeros(np.shape(text[0])[0])
        text_mask[:text[1]] = 1
        # print("audio shape:",audio[0].shape)
        # print("text shape:", text[0].shape)
        # print("maks shape",mask.shape)
        # print("labels shape",labels.shape)
        # print("seqlen",text[1])
        labels=functional.one_hot(torch.from_numpy(np.array(label).astype(np.int64)), num_classes=label_classes)

        return audio[0], audio[0].type(torch.FloatTensor), text[0], audio_mask,text_mask, label, audio[1], text[1],vid

    def __len__(self):
        if self.train_or_test == 'train':
            return len(self.trainVid)
        if self.train_or_test == 'test':
            return len(self.testVid)

