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
    if len(dp['key']) == 0:
        continue
    if dp['file']['textid'] in path2file:
        path2file[dp['file']['textid']].append(dp)
    else:
        path2file[dp['file']['textid']] = [dp]
names = ['train', 'valid', 'test']
sets = {}
for name in names:
    sets[name] = []
    with open('data/'+name+'/dataset.pkl', 'rb') as f:
        data_list = pickle.load(f)
        for dp_c in data_list:
            r_name = dp_c[3]
            ans = re.findall(r"\\(.*)\.pkl", r_name)[0]
            sets[name].append(ans)
for name in names:
    plist = sets[name]
    temp = []
    for path in plist:
        if path in path2file:
            temp += path2file[path]
    with open('data/'+name+'.pkl', 'wb') as f:
        pickle.dump(temp, f)

