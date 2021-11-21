import torch

import os
import logging
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
# 绘制多分类混淆矩阵
from sklearn.metrics import confusion_matrix
import seaborn as sn

import pickle
import LoadData

torch.set_printoptions(threshold=np.inf)
label_classes = 5
dropout = 0.2
batch_size = 20
max_audio_length = 700
max_text_length = 100
epochs = 150
audio_feature_Dimension = 100
audio_Linear = 100
audio_lstm_hidden = 100
text_embedding_Dimension = 100
Bert_text_embedding_Dimension = 768
Fbank_Dimension=80
text_Linear = 100
text_lstm_hidden = 100
mix_lstm_hidden = 100
gru_hidden = 50
attention_weight_num = 100
attention_head_num = 1
bidirectional = 2  # 2表示双向LSTM,1表示单向

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt


def plot_matrix(matrix):
    labels_order = ['hap', 'sad', 'neu', 'ang']
    # labels_order = ['1', '2', '3', '4', '5']
    # 利用matplot绘图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels_order)
    ax.set_yticklabels([''] + labels_order)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    for x in range(len(matrix)):
        for y in range(len(matrix)):
            plt.annotate(matrix[y, x], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    return plt

class DIDIlike_fusion(nn.Module):
    def __init__(self, fusion="ADD"):
        super(DIDIlike_fusion, self).__init__()
        self.fusion = fusion
        # 串联concat

        self.Concat2_Linear = torch.nn.Linear(2 * label_classes, label_classes,
                                              bias=False)

        self.Attention1 = torch.nn.MultiheadAttention(audio_feature_Dimension, attention_head_num, dropout=0.2,
                                                      bias=True,
                                                      add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
        self.Attention2 = torch.nn.MultiheadAttention(attention_weight_num, attention_head_num, dropout=0.2,
                                                      bias=True,
                                                      add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
        self.Attention3 = torch.nn.MultiheadAttention(attention_weight_num, attention_head_num, dropout=0.2,
                                                bias=True,
                                                add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
 
        # 特征直接相加
        self.Bert_Text_Linear = torch.nn.Linear(Bert_text_embedding_Dimension, text_Linear, bias=True)
        self.Fbank_Linear = torch.nn.Linear(Fbank_Dimension, audio_Linear, bias=True)
        self.Classify_Linear = torch.nn.Linear(100, label_classes, bias=True)
        self.Softmax = torch.nn.Softmax(dim=1)
        self.GRU_text = torch.nn.GRU(input_size=audio_feature_Dimension , hidden_size=gru_hidden, num_layers=1,
                                bidirectional=True)
        self.GRU_audio = torch.nn.GRU(input_size=audio_feature_Dimension , hidden_size=gru_hidden, num_layers=1,
                                     bidirectional=True)
        self.GRU_fusion = torch.nn.GRU(input_size=audio_feature_Dimension+attention_weight_num, hidden_size=gru_hidden, num_layers=1,
                                      bidirectional=True)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, Audio_Features, Texts_Embedding, Audio_Seqlen,Text_Seqlen, Audio_Mask,Text_Mask):
        input_text = self.Bert_Text_Linear(Texts_Embedding)
        input_audio =  self.Fbank_Linear(Audio_Features)
        ###################将两个特征分别通过GRU获得序列######################
        #获得text过lstm的输出
        text_Padding = torch.nn.utils.rnn.pack_padded_sequence(input_text, Text_Seqlen, batch_first=True, enforce_sorted=False)
        # LSTM_Out、Text_LSTM_Out的维度分别是audio序列长度，batchsize，特征维度。
        GRU_text_Out, _ = torch.nn.utils.rnn.pad_packed_sequence(self.GRU_text(text_Padding)[0])

        #获得audio过lstm的输出
        audio_Padding = torch.nn.utils.rnn.pack_padded_sequence(input_audio, Audio_Seqlen, batch_first=True, enforce_sorted=False)
        # LSTM_Out、Text_LSTM_Out的维度分别是audio序列长度，batchsize，特征维度。
        GRU_audio_Out, _ = torch.nn.utils.rnn.pad_packed_sequence(self.GRU_audio(audio_Padding)[0])

        MinMask_audio = Audio_Mask[:, :GRU_audio_Out.shape[0]]
        MinMask_text = Text_Mask[:, :GRU_text_Out.shape[0]]
        Contribute_audio = self.dropout(GRU_audio_Out)
        Contribute_text = self.dropout(GRU_text_Out)
        ################两种设置QKV的方式###########################
        # print(Contribute_text.shape)
        # print(Contribute_audio.shape)
        #if self.fusion=="AT":####其实并不是AT，只是不想改代码了
        attention_out1,_=self.Attention1(Contribute_text, Contribute_audio, Contribute_audio,key_padding_mask=(1 - MinMask_audio))
        # if self.fusion=="ADD":####其实并不是ADD，只是不想改代码了
        #     attention_out1,_=self.Attention1(Contribute_text, Contribute_audio, Contribute_audio, key_padding_mask=(1 - MinMask_audio))
        ###############再通过一层GRU###########################
        attention_gru=torch.cat([attention_out1,Contribute_text],dim=2)
        GRUfusion_out,_=self.GRU_fusion(attention_gru)

        #attention_out2,_=self.Attention2(GRUfusion_out, GRUfusion_out, GRUfusion_out,key_padding_mask=(1 - MinMask_audio))
        # print(attention_out1.shape)
        ###############Max_pooling###########################
        Pooling=torch.max(GRUfusion_out,0).values
        # print(Pooling.shape)
        ###############FC层后得到结果#######################
        Emotion_Output=self.Classify_Linear(Pooling)
        Emotion_Predict = self.Softmax(Emotion_Output)
        return Emotion_Predict


def train_and_test_inadvance_middle_fusion(train_loader, test_loader, model, criterion, optimizer, num_epochs, savefile=None):
    Best_Valid = 0

    Loss_Function = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(num_epochs):

        for i, features in enumerate(train_loader):
            video_train, audio_train, text_train, train_audio_mask, train_text_mask,\
            train_label, seqlen_audio_train, seqlen_text_train,vid= features
  
            train_text_mask = train_text_mask.to(torch.int)
            train_audio_mask = train_audio_mask.to(torch.int)
            audio_train = audio_train.to(device)
            train_label = train_label.to(device)
            text_train = text_train.to(device)
            outputs = model.forward(audio_train, text_train, seqlen_audio_train,\
                seqlen_text_train, train_audio_mask,train_text_mask)
            optimizer.zero_grad()
            loss = Loss_Function(outputs, train_label)
            total_loss = torch.sum(loss, dtype=torch.float)

            total_loss.backward()
            optimizer.step()
            torch.save(model,'tmp.pt')

        with torch.no_grad():
            test_model = torch.load('tmp.pt')
            test_model.eval()

            confusion_Ypre = []
            confusion_Ylabel = []
            confusion_TrainYlabel = []
            correct = 0
            total = 0
            for i, features in enumerate(test_loader):
            # for i, features in enumerate(train_loader):
                video_test, audio_test, text_test, test_audio_mask, test_text_mask,test_label,\
                     seqlen_audio_test,seqlen_text_test,vid = features
                test_audio_mask = test_audio_mask.to(torch.int)
                test_text_mask = test_text_mask.to(torch.int)
                audio_test = audio_test.to(device)
                test_label = test_label.to(device)
                text_test = text_test.to(device)

                original_outputs = test_model.forward(audio_test, text_test, seqlen_audio_test,\
                                                seqlen_text_test, test_audio_mask,test_text_mask)

                total+=np.sum(np.shape(test_label))
                _, predict = torch.max(original_outputs, 1)
                correct += (predict == test_label).sum()


        if correct / total > Best_Valid:
            torch.save(model,'best.pt')
            ####################总的混淆矩阵##################
            Best_Valid = correct / total
            matrix = confusion_matrix(confusion_Ylabel, confusion_Ypre)
            total_num = np.sum(matrix, axis=1)
            acc_matrix = np.round(matrix / total_num[:, None], decimals=2)
        print('Epoch: %d/%d; total utterance: %d ; correct utterance: %d ; Acc: %.2f%%' % (epoch + 1, num_epochs,total,correct, 100 * (correct / total)))
            # plot_matrix(acc_matrix).show()
    print("Best Valid Accuracy: %0.2f%%" % (100 * Best_Valid))
    if savefile != None:
        np.savez(savefile, matrix=acc_matrix, ACC=Best_Valid)
