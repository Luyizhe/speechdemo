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
text_Linear = 100
text_lstm_hidden = 100
mix_lstm_hidden = 100
gru_hidden = 50
attention_weight_num = 100
attention_head_num = 1
bidirectional = 2  # 2表示双向LSTM,1表示单向

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiModel(nn.Module):
    def __init__(self):
        super(MultiModel, self).__init__()
        # self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        # self.vocab_size = vocab_size
        # self.dense = nn.Linear(self.hidden_size, vocab_size)
        # self.state = None
        self.Audio_LSTM = torch.nn.LSTM(input_size=audio_feature_Dimension, hidden_size=audio_lstm_hidden, num_layers=1,
                                        bidirectional=True)
        self.Text_LSTM = torch.nn.LSTM(input_size=text_embedding_Dimension, hidden_size=text_lstm_hidden, num_layers=1,
                                       bidirectional=True)
        self.Mix_LSTM = torch.nn.LSTM(
            input_size=(attention_head_num * audio_lstm_hidden + text_lstm_hidden) * bidirectional,
            hidden_size=mix_lstm_hidden, num_layers=1, bidirectional=True)
        self.Linear = torch.nn.Linear(mix_lstm_hidden * 2, label_classes, bias=True)

        # attention权重的维度分别是，head数，特征维度，输出维度
        self.audio_attention_weight = torch.tensor(
            np.random.normal(0, 0.01,
                             size=(attention_head_num, audio_lstm_hidden * bidirectional, attention_weight_num)),
            device=device, dtype=torch.float32)
        self.text_attention_weight = torch.tensor(
            np.random.normal(0, 0.01,
                             size=(attention_head_num, text_lstm_hidden * bidirectional, attention_weight_num)),
            device=device, dtype=torch.float32)
        # attention偏置的维度分别是，head数，输出维度
        self.attention_bias = torch.tensor(
            np.random.normal(0, 0.01, size=(attention_head_num, attention_weight_num)),
            device=device, dtype=torch.float32)
        # attention层结合之后收束成一个值，，
        self.linear_1_weight = torch.tensor(
            np.random.normal(0, 0.01, size=(attention_weight_num, 1)),
            device=device, dtype=torch.float32)

    def forward(self, Audio_Features, Texts_Embedding):
        # print(Texts_Embedding[0].shape)
        # Texts_Embedding[0] = torch.reshape(torch.cat(Texts_Embedding[0]),
        #                                    (batch_size, max_text_length, text_embedding_Dimension))
        # 为了batch统一长度后标记原长度。
        Audio_Padding = torch.nn.utils.rnn.pack_padded_sequence(Audio_Features[0], Audio_Features[1],
                                                                batch_first=True, enforce_sorted=False)
        # Audio_LSTM_Out、Text_LSTM_Out的维度分别是audio序列长度，batchsize，特征维度。
        Audio_LSTM_Out, _ = torch.nn.utils.rnn.pad_packed_sequence(self.Audio_LSTM(Audio_Padding)[0])
        # 为了batch统一长度后标记原长度。
        TextPadding = torch.nn.utils.rnn.pack_padded_sequence(Texts_Embedding[0], Texts_Embedding[1],
                                                              batch_first=True, enforce_sorted=False)
        # Audio_LSTM_Out、Text_LSTM_Out的维度分别是audio序列长度，batchsize，特征维度。
        Text_LSTM_Out, _ = torch.nn.utils.rnn.pad_packed_sequence(self.Text_LSTM(TextPadding)[0])
        # us_i、uh_i的维度分别是串长，head数，batchsize，输出维度。
        us_i = torch.matmul(Audio_LSTM_Out[:, None, :, :], self.audio_attention_weight)
        uh_i = torch.matmul(Text_LSTM_Out[:, None, :, :], self.text_attention_weight)
        # # us_i、uh_i的维度分别是batchsize，head数，串长，输出维度。
        # us_i = us_i.permute([2, 1, 0, 3])
        # uh_i = uh_i.permute([2, 1, 3, 0])
        # a_ji=torch.matmul(us_i,uh_i)/np.sqrt(attention_weight_num)#为了消去0对softmax的影响
        # softmaxed_a_ji = F.softmax(a_ji, dim=-1)

        # 过一个非线性层，维度不变。
        a_ji_tmp = torch.tanh(us_i[:, None, :, :, :] + uh_i + self.attention_bias[:, None, :])
        # VT*tanh,输出维度为：audio序列长度，text序列长度，attention head数，batch_size,多出的1维
        a_ji = torch.matmul(a_ji_tmp, self.linear_1_weight)
        # softmax归一化
        alpha = F.softmax(a_ji, dim=0)
        # print(alpha[:,0,0,0])
        # 转置后为：attention head，batchsize，text序列长度，audio序列长度
        alpha = torch.squeeze(alpha).permute([2, 3, 1, 0])
        # 转置后为：batchsize，audio序列长度，特征维度
        Audio_LSTM_Out = Audio_LSTM_Out.permute([1, 0, 2])
        # 矩阵乘完变成：attention head，batchsize，text序列长度，特征维度
        Attention_Out = torch.matmul(alpha, Audio_LSTM_Out)
        # 转置并reshape后变成：attention后特征维度，batchsize，text序列长度
        Attention_Out = Attention_Out.permute([0, 3, 1, 2]).reshape(
            attention_head_num * audio_lstm_hidden * bidirectional,
            batch_size, -1)
        # 将Text_LSTM_Out和Attention_Out统一成：batchsize，text序列长度，特征维度，再将两者拼接
        Text_LSTM_Out = Text_LSTM_Out.permute([1, 0, 2])
        Attention_Out = Attention_Out.permute([1, 2, 0])
        Final_LSTM_In = torch.cat((Text_LSTM_Out, Attention_Out), dim=2)
        # Mix_LSTM_Out的维度为：batchsize，text序列长度，特征维度。
        Mix_LSTM_Out, _ = self.Mix_LSTM(Final_LSTM_In)
        # Pooling之后的维度为:batchsize，特征维度
        Pooling_Out = torch.squeeze(torch.max(Mix_LSTM_Out, dim=1)[0])
        Linear_out = self.Linear(Pooling_Out)
        Model_out = F.softmax(Linear_out, dim=1)
        return Model_out


class DialogueModal(nn.Module):
    def __init__(self):
        super(DialogueModal, self).__init__()

        self.Audio_Linear = torch.nn.Linear(audio_feature_Dimension, audio_Linear, bias=True)
        self.Text_Linear = torch.nn.Linear(text_embedding_Dimension, text_Linear, bias=True)
        # 串联concat
        self.Concat1_Linear = torch.nn.Linear(text_embedding_Dimension, audio_Linear,
                                              bias=False)
        self.Omega_f = torch.randn(text_embedding_Dimension, 1, requires_grad=True)

        self.AT_Linear = torch.nn.Linear(audio_Linear, 1, bias=True)
        self.AT_softmax = torch.nn.Softmax(dim=2)
        self.GRU = torch.nn.GRU(input_size=audio_Linear, hidden_size=gru_hidden, num_layers=1,
                                bidirectional=True)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.Attention = torch.nn.MultiheadAttention(attention_weight_num, attention_head_num, dropout=0.2, bias=True,
                                                     add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
        self.Q_attention_weight = torch.tensor(
            np.random.normal(0, 0.01,
                             size=(attention_head_num, gru_hidden * bidirectional,
                                   int(attention_weight_num / attention_head_num))),
            device=device, dtype=torch.float32)
        self.K_attention_weight = torch.tensor(
            np.random.normal(0, 0.01,
                             size=(attention_head_num, gru_hidden * bidirectional,
                                   int(attention_weight_num / attention_head_num))),
            device=device, dtype=torch.float32)
        self.V_attention_weight = torch.tensor(
            np.random.normal(0, 0.01,
                             size=(attention_head_num, gru_hidden * bidirectional,
                                   int(attention_weight_num / attention_head_num))),
            device=device, dtype=torch.float32)
        self.attention_bias = torch.tensor(
            np.random.normal(0, 0.01, size=(attention_head_num, int(attention_weight_num / attention_head_num))),
            device=device, dtype=torch.float32)
        self.linear_1_weight = torch.tensor(
            np.random.normal(0, 0.01, size=(int(attention_weight_num / attention_head_num), 1)),
            device=device, dtype=torch.float32)
        self.Emotion_Classify = torch.nn.Linear(attention_weight_num, label_classes, bias=True)
        self.Emotion_softmax = torch.nn.Softmax(dim=2)

    def forward(self, Audio_Features, Texts_Embedding, seqlen):
        # 横向合并，batch,length,embedding变成batch,length,embedding,2
        Concat = torch.cat([Audio_Features[:, :, None, :], Texts_Embedding[:, :, None, :]], 2)
        u_cat = self.Concat1_Linear(Concat)
        NonLinear = self.dropout(torch.tanh(u_cat))
        alpha_fuse = torch.matmul(NonLinear, self.Omega_f)
        alpha_fuse = alpha_fuse.squeeze(3)
        AT_fusion = torch.matmul(u_cat.permute([0, 1, 3, 2]), alpha_fuse[:, :, :, None]).squeeze(dim=3)
        # Audio_Linear_Out = self.Audio_Linear(Audio_Features)
        # Audio_Contribute = self.dropout(torch.tanh(Audio_Linear_Out))
        # Text_Linear_Out = self.Text_Linear(Texts_Embedding)
        # Text_Contribute = self.dropout(torch.tanh(Text_Linear_Out))
        # Audio_AT = self.AT_Linear(Audio_Contribute)
        # Text_AT = self.AT_Linear(Text_Contribute)
        # u_cat = torch.cat((torch.unsqueeze(Audio_Linear_Out, dim=3), torch.unsqueeze(Text_Linear_Out, dim=3)), 3)
        #
        # AT_fusion_weights = torch.cat((Audio_AT, Text_AT), 2)
        # AT_attention = self.AT_softmax(AT_fusion_weights)
        # AT_fusion = torch.matmul(u_cat, alpha_fuse.unsqueeze(dim=3)).squeeze(dim=3)

        # 串联fusion
        # Fusion = torch.cat([Audio_Features, Texts_Embedding], 2)
        # AT_Linear = self.Concat_Linear(Fusion)
        # AT_fusion = self.dropout(torch.tanh(AT_Linear))

        # 为了batch统一长度后标记原长度。
        AT_fusion_Padding = torch.nn.utils.rnn.pack_padded_sequence(AT_fusion, seqlen,
                                                                    batch_first=True, enforce_sorted=False)

        # AT_fusion_Padding = torch.nn.utils.rnn.pack_padded_sequence(Text_Contribute, seqlen,
        #                                                              batch_first=True, enforce_sorted=False)
        # Audio_LSTM_Out、Text_LSTM_Out的维度分别是audio序列长度，batchsize，特征维度。

        SA_GRU_GRU, _ = torch.nn.utils.rnn.pad_packed_sequence(self.GRU(AT_fusion_Padding)[0])

        # uq_i = torch.matmul(SA_GRU_GRU[:, None, :, :], self.K_attention_weight)
        # uk_i = torch.matmul(SA_GRU_GRU[:, None, :, :], self.Q_attention_weight)
        # uv_i = torch.matmul(SA_GRU_GRU[:, None, :, :], self.V_attention_weight)
        #
        # a_ji_tmp = torch.tanh(uq_i[:, None, :, :, :] + uk_i + self.attention_bias[:, None, :])
        #
        # a_ji = torch.matmul(a_ji_tmp, self.linear_1_weight)
        # # softmax归一化
        # alpha = F.softmax(a_ji, dim=0)
        # # 转置后为：attention head，batchsize，text序列长度，audio序列长度
        # alpha = torch.squeeze(alpha).permute([2, 3, 1, 0])
        #
        # # 转置后为：batchsize，audio序列长度，特征维度
        # SA_GRU_GRU = SA_GRU_GRU.permute([1, 0, 2])
        # print(SA_GRU_GRU.shape)
        # exit()
        # uv_i = uv_i.permute([1, 2, 0, 3])
        #
        # # 矩阵乘完变成：attention head，batchsize，text序列长度，特征维度
        # Attention_Out = torch.matmul(alpha, uv_i)
        # # Attention_Out = torch.matmul(alpha, SA_GRU_GRU)
        #
        # # 转置并reshape后变成：attention后特征维度，batchsize，text序列长度
        # Attention_Out = Attention_Out.permute([0, 3, 1, 2]).reshape(
        #     attention_weight_num,
        #     Texts_Embedding[0].shape[0], -1)

        mask_key = torch.BoolTensor(SA_GRU_GRU.shape[0:2])
        mask_seq = torch.BoolTensor(SA_GRU_GRU.shape[1] * attention_head_num, SA_GRU_GRU.shape[0], SA_GRU_GRU.shape[0])
        mask_key[:, :] = 1
        seqlen = seqlen.to(torch.int)
        # mask_seq[:, :, :] = 1
        for i in range(SA_GRU_GRU.shape[1]):
            mask_key[:seqlen[i], i] = 0
        # for i in range(SA_GRU_GRU.shape[1]):
        #     for j in range (attention_head_num):
        #         mask_seq[i+j, :Texts_Embedding[1][i], :Texts_Embedding[1][i]] = 0

        mask_key = mask_key.permute([1, 0])

        SA_GRU_SA, _ = self.Attention(SA_GRU_GRU, SA_GRU_GRU, SA_GRU_GRU, key_padding_mask=mask_key)

        Emotion_classify = self.Emotion_Classify(SA_GRU_SA)
        # Emotion_classify = self.Emotion_Classify(SA_GRU_SA.permute([2, 1, 0]))
        Emotion_classify_predict = self.Emotion_softmax(Emotion_classify)

        return Emotion_classify_predict


class Multilevel_Multiple_Attentions(nn.Module):
    def __init__(self, modal="text", fusion="AT_fusion"):
        super(Multilevel_Multiple_Attentions, self).__init__()
        self.modal = modal
        self.fusion = fusion
        # 串联concat
        self.Concat2_Linear_Bert = torch.nn.Linear(Bert_text_embedding_Dimension + audio_Linear, audio_Linear,
                                                   bias=False)
        self.Concat2_Linear = torch.nn.Linear(text_embedding_Dimension + audio_Linear, audio_Linear,
                                              bias=False)
        # 并联concat
        self.Concat1_Linear = torch.nn.Linear(text_embedding_Dimension, audio_Linear,
                                              bias=False)
        self.Omega_f = torch.normal(mean=torch.full((text_embedding_Dimension, 1), 0.0),
                                    std=torch.full((text_embedding_Dimension, 1), 0.01))
        # 特征直接相加
        # self.ADD= torch.nn.Linear(text_embedding_Dimension, audio_Linear,bias=False)

        self.Audio_Linear = torch.nn.Linear(audio_feature_Dimension, audio_Linear, bias=True)
        self.Text_Linear = torch.nn.Linear(text_embedding_Dimension, text_Linear, bias=True)
        self.Bert_Text_Linear = torch.nn.Linear(Bert_text_embedding_Dimension, text_Linear, bias=True)
        self.Linear = torch.nn.Linear(attention_weight_num, 100, bias=True)
        self.Classify_Linear = torch.nn.Linear(100, label_classes, bias=True)
        self.Softmax = torch.nn.Softmax(dim=2)
        self.GRU = torch.nn.GRU(input_size=audio_Linear, hidden_size=gru_hidden, num_layers=1,
                                bidirectional=True)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.Attention = torch.nn.MultiheadAttention(attention_weight_num, attention_head_num, dropout=0.2, bias=True)

    def forward(self, Audio_Features, Texts_Embedding, Seqlen, Mask):
        # print(Texts_Embedding[0].shape)
        # Texts_Embedding[0] = torch.reshape(torch.cat(Texts_Embedding[0]),
        #                                    (batch_size, max_text_length, text_embedding_Dimension))

        # Audio_Linear_Out = self.Audio_Linear(Audio_Features[0])
        # Audio_Contribute = self.dropout(torch.tanh(Audio_Linear_Out))

        if self.modal == "text":
            # 非Bert的embedding
            #input = Texts_Embedding
            #Bert的embedding
            input = self.Bert_Text_Linear(Texts_Embedding)
        if self.modal == "audio":
            input = Audio_Features


        if self.modal == "multi":
            if self.fusion == "AT_fusion":
                # 并联的方式（AT-fusion）

                # Concat = torch.cat([Audio_Features[:, :, None, :], Texts_Embedding[:, :, None, :]], 2)
                # Bert的文本

                tmp = self.Bert_Text_Linear(Texts_Embedding)

                Concat = torch.cat([Audio_Features[:, :, None, :], tmp[:, :, None, :]], 2)
                u_cat = self.Concat1_Linear(Concat)

            elif self.fusion == "Concat":
                # 串联

                Concat = torch.cat([Audio_Features[:, :, None, :], Texts_Embedding[:, :, None, :]], 3)
                u_cat = self.Concat2_Linear_Bert(Concat)
                # u_cat = self.Concat2_Linear(Concat)
            if self.fusion != "ADD":
                NonLinear = self.dropout(torch.tanh(u_cat))
                alpha_fuse = torch.matmul(NonLinear, self.Omega_f)
                alpha_fuse = alpha_fuse.squeeze(3)
                normalized_alpha = self.Softmax(alpha_fuse)
                input = torch.matmul(u_cat.permute([0, 1, 3, 2]), normalized_alpha[:, :, :, None]).squeeze(dim=3)
            if self.fusion == "ADD":
                # ADD的方式
                # 非Bert
                # input = Audio_Features[:, :, :] + Texts_Embedding[:, :, :]
                # Bert
                input = Audio_Features[:, :, :] + self.Bert_Text_Linear(Texts_Embedding[:, :, :])
        # 为了batch统一长度后标记原长度。
        Audio_Padding = torch.nn.utils.rnn.pack_padded_sequence(input, Seqlen,
                                                                batch_first=True, enforce_sorted=False)
        # Audio_LSTM_Out、Text_LSTM_Out的维度分别是audio序列长度，batchsize，特征维度。
        Audio_GRU_Out, _ = torch.nn.utils.rnn.pad_packed_sequence(self.GRU(Audio_Padding)[0])

        MinMask = Mask[:, :Audio_GRU_Out.shape[0]]
        Audio_Contribute = self.dropout(Audio_GRU_Out)

        Attention_Out, _ = self.Attention(Audio_Contribute, Audio_Contribute, Audio_Contribute,
                                          key_padding_mask=(1 - MinMask))

        Dense1 = torch.tanh(self.Linear(Attention_Out.permute([1, 0, 2])))
        Masked_Dense1 = Dense1 * MinMask[:, :, None]
        Dropouted_Dense1 = self.dropout(Masked_Dense1)
        Emotion_Output = self.Classify_Linear(Dropouted_Dense1.permute([1, 0, 2]))

        Emotion_Predict = self.Softmax(Emotion_Output)


        return Emotion_Predict


def train(train_loader, model, criterion, optimizer, num_epochs):
    loss = 0.0
    for epoch in range(num_epochs):
        for i, features in enumerate(train_loader):
            Audio_Features, Labels, Texts_Embedding = features

            Audio_Features[0] = Audio_Features[0].to(device)
            Labels = Labels.to(device)
            Texts_Embedding[0] = Texts_Embedding[0].to(device)
            outputs = model.forward(Audio_Features, Texts_Embedding)

            optimizer.zero_grad()
            loss = criterion(outputs, Labels)
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, loss.item()))


def test(test_loader, model):
    correct = 0
    total = 0
    for i, features in enumerate(test_loader):
        Audio_Features, Labels, Texts_Embedding = features
        Audio_Features[0] = Audio_Features[0].to(device)
        Labels = Labels.to(device)
        Texts_Embedding[0] = Texts_Embedding[0].to(device)
        outputs = model.forward(Audio_Features, Texts_Embedding)
        # torch.max并不是np.max一个意思，是用以计算sofamax的分类类别的，建议CSDN查一下
        _, predict = torch.max(outputs, 1)
        total += Labels.size(0)
        correct += (predict == Labels).sum()
    print(total, correct)


def train_and_test_Sen(train_loader, test_loader, model, criterion, optimizer, num_epochs):
    loss = 0.0
    for epoch in range(num_epochs):

        for i, features in enumerate(train_loader):
            Audio_Features, Labels, Texts_Embedding = features

            Audio_Features[0] = Audio_Features[0].to(device)

            Labels = Labels.to(device)
            Texts_Embedding[0] = Texts_Embedding[0].to(device)
            outputs = model.forward(Audio_Features, Texts_Embedding)

            optimizer.zero_grad()

            loss = criterion(outputs, Labels)
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, loss.item()))

        with torch.no_grad():
            correct = 0
            total = 0
            for i, features in enumerate(test_loader):
                Audio_Features, Labels, Texts_Embedding = features
                Audio_Features[0] = Audio_Features[0].to(device)
                Labels = Labels.to(device)
                Texts_Embedding[0] = Texts_Embedding[0].to(device)
                outputs = model.forward(Audio_Features, Texts_Embedding)
                # torch.max并不是np.max一个意思，是用以计算sofamax的分类类别的，建议CSDN查一下
                _, predict = torch.max(outputs, 1)
                total += Labels.size(0)
                correct += (predict == Labels).sum()
            print(total, correct)
            print(correct / total)


def train_and_test_Dia(train_loader, test_loader, model, criterion, optimizer, num_epochs):
    loss = 0.0
    Best_Valid = 0
    for epoch in range(num_epochs):
        for i, features in tqdm(enumerate(train_loader)):
            video_train, Audio_Features, Texts_Embedding, train_mask, Labels, seqlen_train = features
            # Audio_Features, Labels, Texts_Embedding = features

            Audio_Features = Audio_Features.to(device)
            Labels = Labels.to(device)
            Texts_Embedding = Texts_Embedding.to(device)

            outputs = model.forward(Audio_Features, Texts_Embedding, seqlen_train)
            Labels = Labels[:, 0:outputs.shape[0]]
            outputs = outputs.permute([1, 2, 0])

            optimizer.zero_grad()

            Loss_Label = torch.argmax(Labels, dim=2)
            loss = criterion(outputs, Loss_Label)

            True_loss = loss * train_mask[:, :loss.shape[1]]
            total_loss = torch.sum(torch.sum(True_loss))
            total_loss.backward()
            optimizer.step()

            # print('Epoch [{}/{}], Loss: {:.4f}'
            #       .format(epoch + 1, num_epochs, total_loss.item()))

        with torch.no_grad():
            correct = 0
            total = 0
            for i, features in enumerate(test_loader):
                # for i, features in enumerate(train_loader):
                video_test, Audio_Features, Texts_Embedding, test_mask, Labels, seqlen_test = features
                # Audio_Features, Labels, Texts_Embedding = features
                Audio_Features = Audio_Features.to(device)
                Labels = Labels.to(device)
                Texts_Embedding = Texts_Embedding.to(device)

                outputs = model.forward(Audio_Features, Texts_Embedding, seqlen_test)
                Labels = Labels[:, 0:outputs.shape[0]]
                outputs = outputs.permute([1, 0, 2])
                # torch.max并不是np.max一个意思，是用以计算sofamax的分类类别的，建议CSDN查一下
                _, predict = torch.max(outputs, 2)
                test_mask = test_mask[:, :predict.shape[1]]
                predict = predict * test_mask
                Labels = torch.argmax(Labels, dim=2)
                Labels = Labels * test_mask
                total += test_mask.sum()
                correct += ((predict == Labels) * test_mask).sum()
            print('Epoch: %d/%d; total utterance: %d ; correct utterance: %d ; Acc: %.2f%%' % (epoch + 1, num_epochs,
                                                                                               total.item(),
                                                                                               correct.item(), 100 * (
                                                                                                       correct / total).item()))
            if correct / total > Best_Valid:
                Best_Valid = correct / total

    print("Best Valid Accuracy: %0.2f%%" % (100 * Best_Valid))


np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt


def plot_matrix(matrix):
    # labels_order = ['hap', 'sad', 'neu', 'ang']
    labels_order = ['1', '2', '3', '4', '5']
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


def train_and_test_Multilevel(train_loader, test_loader, model, criterion, optimizer, num_epochs, savefile=None):
    loss = 0.0
    Best_Valid = 0
    Best_AUC = 0
    # for lab_times in tqdm(range(5)):
    #     time.sleep(5)
    if criterion == "MSELoss":
        FC = torch.nn.Linear(label_classes, 1, bias=True)
        Loss_Function = nn.MSELoss(reduction='none')
    elif criterion == "CrossEntropyLoss":
        Loss_Function = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(num_epochs):
        confusion_Ypre = []
        confusion_Ylabel = []
        confusion_TrainYlabel = []
        outputs_total = []
        test_label_total = []
        for i, features in enumerate(train_loader):
            video_train, audio_train, text_train, train_mask, train_label, seqlen_train = features
            train_mask = train_mask.to(torch.int)

            audio_train = audio_train.to(device)
            train_label = train_label.to(device)
            text_train = text_train.to(device)

            outputs = model.forward(audio_train, text_train, seqlen_train, train_mask)

            train_label = train_label[:, 0:outputs.shape[0]]
            outputs = outputs.permute([1, 2, 0])
            train_label = train_label.permute([0, 2, 1])
            Loss_Label = torch.argmax(train_label, dim=1)
            optimizer.zero_grad()

            if criterion == "MSELoss":
                outputs = outputs.permute([0, 2, 1])
                final = FC(outputs).squeeze(2)
                loss = Loss_Function(final, Loss_Label.to(torch.float))
                mask_extend = train_mask[:, :loss.shape[1]]
                True_loss = loss * mask_extend
                total_loss = torch.sum(True_loss, dtype=torch.float)
            elif criterion == "CrossEntropyLoss":
                loss = Loss_Function(outputs, Loss_Label)
                True_loss = loss * train_mask[:, :loss.shape[1]]
                total_loss = torch.sum(True_loss, dtype=torch.float)

            total_loss.backward()
            optimizer.step()
            for i in range(Loss_Label.shape[0]):
                confusion_TrainYlabel.extend(Loss_Label[i][:seqlen_train[i]].numpy())
            torch.save(model,"retest.pt")

        with torch.no_grad():
            test_model=torch.load("retest.pt")
            #test_model.eval()
            correct = 0
            total = 0
            RMSE=0
            best_RMSE=1000000
            for i, features in enumerate(test_loader):
            #for i, features in enumerate(train_loader):
                video_test, audio_test, text_test, test_mask, test_label, seqlen_test = features
                test_mask = test_mask.to(torch.int)
                audio_test = audio_test.to(device)
                test_label = test_label.to(device)
                text_test = text_test.to(device)

                original_outputs = test_model.forward(audio_test, text_test, seqlen_test, test_mask)
                if criterion=="MSELoss":
                    outputs_original=FC(original_outputs).squeeze(2)
                    predict=outputs_original.permute([1, 0])
                elif criterion=="CrossEntropyLoss":
                    outputs=original_outputs
                    outputs_original = outputs.permute([1, 0, 2])
                    _, predict = torch.max(outputs_original, 2)
                test_label_original = test_label[:, :predict.shape[1]]
                # torch.max并不是np.max一个意思，是用以计算sofamax的分类类别的，建议CSDN查一下

                test_mask = test_mask[:, :predict.shape[1]]
                predict = predict * test_mask
                test_label = torch.argmax(test_label_original, dim=2)
                test_label = test_label * test_mask
                total += test_mask.sum()

                # matrix = confusion_matrix(test_label[0], predict[0])
                # total_num = np.sum(matrix, axis=0)
                # acc = np.round(matrix / total_num[None, :], decimals=2)
                correct += ((predict == test_label) * test_mask).sum()
                if criterion=="MSELoss":
                    for i in range(predict.shape[0]):
                        predict_line=predict[:, :seqlen_test[i]].reshape(-1)
                        test_label_line=test_label[:, :seqlen_test[i]].reshape(-1)
                        outputs_total.extend(test_label_line)
                        RMSE=RMSE+torch.sum((predict_line-test_label_line)**2)
                elif criterion=="CrossEntropyLoss":
                    for i in range(predict.shape[0]):
                        confusion_Ypre.extend(predict[i][:seqlen_test[i]].numpy())
                        confusion_Ylabel.extend(test_label[i][:seqlen_test[i]].numpy())
                        outputs_total.extend(outputs_original[:, :seqlen_test[i], :].reshape(-1))
                        test_label_total.extend(test_label_original[:, :seqlen_test[i], :].reshape(-1))
            if criterion == "MSELoss":
                RMSE=RMSE/len(outputs_total)
                RMSE=np.sqrt(RMSE)
        # if criterion=="CrossEntropyLoss":
        #     AUC = roc_auc_score(test_label_total, outputs_total)
        # elif criterion=="MSELoss":
        #     AUC=0

            # if criterion == "MSELoss":
            #     print('Epoch: %d/%d; total utterance: %d ; correct utterance: %d ; Auc: %.2f' % (epoch + 1, num_epochs,
            #                                                                                      total.item(),
            #                                                                                      correct.item(),
            #                                                                                      AUC))
            # elif criterion == "CrossEntropyLoss":
            #     print(
            #         'Epoch: %d/%d; total utterance: %d ; correct utterance: %d ; Acc: %.2f%%' % (epoch + 1, num_epochs,
            #                                                                                      total.item(),
            #                                                                                      correct.item(), 100 * (
            #                                                                                              correct / total).item()))
        if RMSE<best_RMSE:
            best_RMSE=RMSE
        # if AUC > Best_AUC:
        #     Best_AUC = AUC
        #     matrix = confusion_matrix(confusion_Ylabel, confusion_Ypre)
        #     total_num = np.sum(matrix, axis=0)
        #     acc_matrix = np.round(matrix / total_num[None, :], decimals=2)
        if correct / total > Best_Valid:
            Best_Valid = correct / total
            matrix = confusion_matrix(confusion_Ylabel, confusion_Ypre)
            total_num = np.sum(matrix, axis=1)
            acc_matrix = np.round(matrix / total_num[:, None], decimals=2)
            # plot_matrix(acc).show()


        # print(confusion_TrainYlabel.count(4))
        # exit()

        # print('Epoch [{}/{}], Loss: {:.4f}'
        #       .format(epoch + 1, num_epochs, total_loss.item()))
    if criterion == "MSELoss":
        print("Best Valid RMSE: %0.2f" % (RMSE))
        np.savez(savefile, RMSE=RMSE)
    elif criterion == "CrossEntropyLoss":
        print("Best Valid Accuracy: %0.2f%%" % (100 * Best_Valid))
        if savefile != None:
            #np.savez(savefile, matrix=acc_matrix,AUC_label=test_label_total,AUC_outputs=outputs_total)
            np.savez(savefile, matrix=acc_matrix)
