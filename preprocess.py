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
type_dict = pickle.load(open('./data/pmid_type_dict.pkl', 'rb'))
type2ids = {}
for (key, value) in type_dict.items():
    if value in type2ids:
        type2ids[value].append(key)
    else:
        type2ids[value] = [key]
for key in type2ids.keys():
    print(key)
    print(len(type2ids[key]))
diseases_list = list(type2ids.keys())
total = len(diseases_list)
if not os.path.exists('./data/train_valid_test_diseases.pkl'):
    import random
    random.seed(0)
    random.shuffle(diseases_list)
    train_diseases = diseases_list[:int(total/10*7)]
    valid_diseases = diseases_list[int(total/10*7):int(total/10*8)]
    test_diseases = diseases_list[int(total/10*8):]
    train_valid_test = [train_diseases, valid_diseases, test_diseases]
    pickle.dump(train_valid_test, open('./data/train_valid_test_diseases.pkl', 'wb'))
else:
    train_valid_test = pickle.load(open('./data/train_valid_test_diseases.pkl', 'rb'))
    train_diseases = train_valid_test[0]
    valid_diseases = train_valid_test[1]
    test_diseases = train_valid_test[2]
train_data = []
for disease in train_diseases:
    train_data += type2ids[disease]
test_data = []
for disease in test_diseases:
    test_data += type2ids[disease]
valid_data = []
for disease in valid_diseases:
    valid_data += type2ids[disease]
for name, d_set in zip(names, [train_data, valid_data, test_data]):
    sets[name] = []
    for dp_c in d_set:
        sets[name].append(dp_c)
for name in names:
    plist = sets[name]
    temp = []
    for path in plist:
        if path in path2file:
            temp += path2file[path]
    with open(config.data_file.replace('.pkl', '_%s_dataset_raw.pkl' %(name)), 'wb') as f:
        pickle.dump(temp, f)

