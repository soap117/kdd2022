import os
import pickle
import re
from config import config
if os.path.exists(config.data_file):
    with open(config.data_file, 'rb') as f:
        my_data = pickle.load(f)
path2file = {}
def isChinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff' or ch == ' ' or ch == '_' or ch == '-':
            return True
    return False
for dp in my_data:
    try:
        if dp['textid'] in path2file:
            path2file[dp['textid']].append(dp)
        else:
            path2file[dp['textid']] = [dp]
    except:
        if dp['file']['textid'] in path2file:
            path2file[dp['file']['textid']].append(dp)
        else:
            path2file[dp['file']['textid']] = [dp]
names = ['train', 'valid', 'test']
sets = {}
import json
dataset = json.load(open('./data/dataset_new_3.json', 'r', encoding='utf-8'))
total = len(dataset)
train_data = dataset[:int(total/10*8)]
test_data = dataset[int(total/10*8):int(total/10*9)]
valid_data = dataset[int(total/10*9):]
for name, d_set in zip(names, [train_data, valid_data, test_data]):
    sets[name] = []
    for dp_c in d_set:
        sets[name].append(dp_c['textid'])
for name in names:
    plist = sets[name]
    temp = []
    for path in plist:
        if path in path2file:
            temp += path2file[path]
    with open(config.data_file.replace('.pkl', '_%s_dataset_raw.pkl' %(name)), 'wb') as f:
        pickle.dump(temp, f)

