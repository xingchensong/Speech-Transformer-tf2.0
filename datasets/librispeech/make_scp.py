import sys
from os.path import join, basename
from glob import glob
import numpy as np

data_path = 'E:\corpus_en\LibriSpeech'
large = False
medium = False

parts = ['train-clean-100', 'dev-clean', 'dev-other',
             'test-clean', 'test-other']
if large:
    parts += ['train-clean-360', 'train-other-500']
elif medium:
    parts += ['train-clean-360']

# parts = ['dev-clean']

# NOTE:
############################################################
# [character]
# 26 alphabets(a-z), space(_), apostorophe(')
# = 30 labels

# [character_capital_divide]
# - 100h
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 19 special double-letters, apostorophe(')
# 74 labels

# - 460h
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 24 special double-letters, apostorophe(')
# = 79 labels

# - 960h
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 24 special double-letters, apostorophe(')
# = 79 labels

# [word]
# - 100h
# Original: 33798 labels + OOV
# - 460h
# Original: 65987 labels + OOV
# - 960h
# Original: 89114 labels + OOV
#
# [utterances]
# train-clean-100:   28539 utterances
# dev-clean:         2703 utterances
# dev-other:         2864 utterances
# test-clean:        2620 utterances
# test-other:        2939 utterances
############################################################


########################################
# uttid to utt
########################################
uttid2utt = {}
for part in parts:
    trans_paths = [p for p in glob(join(data_path, part, '*/*/*.trans.txt'))]
    vocab = set()
    num_utt = 0
    for trans_path in trans_paths:
        with open(trans_path,'r') as f:
            for line in f.readlines():
                num_utt += 1
                pair = line.strip().split(' ',maxsplit=1)
                uttid2utt[pair[0]] = pair[1].lower().replace(' ','_')
                text = pair[1].lower()
                for i in range(len(text)):
                    vocab.add(text[i])
    vocab = list(vocab)
    print(part+': '+str(num_utt)+'utterances '+str(len(vocab)) + ' '+str(sorted(vocab)))
    # np.save('config/'+part+'_char.npy',sorted(vocab))
# np.save('data/uttid2utt.npy',uttid2utt) # utt最后还要从字符变成数字形式
# print(uttid2utt)
print('Done uttid to utt')


########################################
# write uttid,uttpath,utt
########################################
for part in parts:
    # part/speaker/book/*.wav
    wav_paths = [p for p in glob(join(data_path, part, '*/*/*.wav'))]
    print(part+': '+str(len(wav_paths))+' utterances')
    with open('data/' + part + '.scp', 'w') as f:
        for wav_path in wav_paths:
            # ex.) wav_path: speaker/book/speaker-book-utt_index.wav
            # print(wav_path)
            temp = wav_path.split('\\')[-5::]
            uttpath = ''
            for i in temp:
                uttpath+=i
                uttpath+='/'
            uttpath = uttpath[:-1]

            uttid = basename(wav_path).split('.')[0]
            utt = uttid2utt[uttid]

            f.write(uttpath + '   ' + utt + '   ' + uttid + '\n')
            # ex.) htk_path: speaker/book/speaker-book-utt_index.htk
print('Done write uttid,uttpath,utt')