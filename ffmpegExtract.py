import os
path=r"E:\NLP\SpeechEmotion\dataset\SIMS\Raw\video\\"
for dir in os.listdir(path):
    vedio_id=path+dir
    for tmp in os.listdir(vedio_id):
        mp4=vedio_id+"\\"+tmp
        # os.system(r"ffmpeg -i %s -acodec copy -vn -y %s.m4a" % (
        # mp4, r"E:\NLP\SpeechEmotion\dataset\SIMS\Raw\audio\\" + dir + "_" + tmp.split(".")[0]))
        os.system(r"ffmpeg -i %s -f wav -ar 16000 %s.wav"%(mp4,r"E:\NLP\SpeechEmotion\dataset\SIMS\Raw\audio\\"+dir+"_"+tmp.split(".")[0]))