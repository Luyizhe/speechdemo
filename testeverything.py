# import sys
# sys.path.append(r"E:\NLP\SpeechEmotion\dataset\CMU-MultimodalSDK")
# from mmsdk import mmdatasdk
# path=r"E:\NLP\SpeechEmotion\dataset\MOSI"
# import os
# import numpy as np
# def avg(intervals: np.array, features: np.array) -> np.array:
# # 虽然没有用到intervals，但是还是要作为参数，不然会报错
#     try:
#         return np.average(features, axis=0)
#     except:
#         return features
#
# DATA_PATH=r"E:\NLP\SpeechEmotion\dataset\CMU_MOSI\cmumosi-aligned\\"
# visual_field = r'CMU_MOSI_Visual_Facet_41.csd'
# acoustic_field = r'CMU_MOSI_COVAREP.csd'
# text_field = r'CMU_MOSI_TimestampedWordVectors.csd'
# label_field = r'CMU_MOSI_Opinion_Labels'
#
# features = [
#     text_field,
#     visual_field,
#     acoustic_field,
# ]
# recipe = {feat: os.path.join(DATA_PATH, feat)  for feat in features}
# dataset = mmdatasdk.mmdataset(recipe)
# print(list(dataset.keys()))
# print("=" * 50)
#
# print(list(dataset[acoustic_field].keys())[:10])  # 视觉模态的前十个key，这里即前十个id
# print("=" * 50)
#
# # 第十五个视频的keys，即那个一个元组
# some_id = list(dataset[acoustic_field].keys())[15]
# print(list(dataset[acoustic_field][some_id].keys()))
# print("=" * 50)
#
# # 看一下时间戳的shape
# print(list(dataset[acoustic_field][some_id]['intervals'].shape))
# # print(list(dataset[visual_field][some_id]['intervals']))
# print("=" * 50)
#
# # 看一下每一个模态的shape
# print(list(dataset[visual_field][some_id]['features'].shape))
# print(list(dataset[text_field][some_id]['features'].shape))
# print(list(dataset[acoustic_field][some_id]['features'].shape))
#
# dataset.align(text_field, collapse_functions=[avg])
#
# some_id = list(dataset[acoustic_field].keys())[15]
# print(list(dataset[visual_field][some_id]['features'].shape))
# print(list(dataset[acoustic_field][some_id]['features'].shape))
# print(list(dataset[text_field][some_id]['features'].shape))
# print("=" * 50)
#
# label_recipe = {label_field: os.path.join(DATA_PATH, label_field + '.csd')}
# dataset.add_computational_sequences(label_recipe, destination=None)
# dataset.align(label_field)
#
# some_id = list(dataset[acoustic_field].keys())[10]
# print(list(dataset.keys()))
# print(list(dataset[acoustic_field].keys())[:100])
# print(list(dataset[visual_field][some_id]['features'].shape))
# print(list(dataset[acoustic_field][some_id]['features'].shape))
# print(list(dataset[text_field][some_id]['features'].shape))
# print(list(dataset[label_field][some_id]['features']))
# print("=" * 50)
# # 不同的模态有不同的time step
# print("Different modalities have different number of time steps!")
# print("=" * 50)

import pickle
import numpy as np
tmp=pickle.load(open(r"E:\NLP\SpeechEmotion\dataset\IEMOCAP_full_release\iemocap_data.pkl","rb"), encoding='latin1')
for id in tmp["train"]["audio"]:
    print(id)
    print(np.shape(id))
    break

# tmp=pickle.load(open(r"E:\NLP\SpeechEmotion\dataset\CMU_MOSI\mosi_data.pkl","rb"), encoding='latin1')
# for id in tmp["train"]["vision"]:
#     print(id)
#     print(np.shape(id))
#     exit()

