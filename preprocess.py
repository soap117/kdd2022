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
fail_count = 0
for dp in my_data:
    new_urls = []
    new_rsecs = []
    new_rpsecs = []
    url_set = set()
    if len(dp['urls'])==0:
        for small_data in dp['file']['contents']:
            for list_anno in small_data['tooltips']:
                for temp in list_anno['sources']:
                    dp['urls'].append(temp['link'])
                    dp['rsecs'].append(temp['reference'])
                    dp['rpsecs'].append([])
    for url, rsec, rpsec in zip(dp['urls'], dp['rsecs'], dp['rpsecs']):
        if url not in url_set:
            title = rpsec[-1]
            if not isChinese(title):
                print(title)
                fail_count += 1
                continue
            new_urls.append(url)
            new_rsecs.append(rsec)
            new_rpsecs.append(rpsec)
            url_set.add(url)
    dp['urls'] = new_urls
    dp['rsecs'] = new_rsecs
    dp['rpsecs'] = new_rpsecs
    path2file[dp['file']['textid']] = dp
    if len(new_rpsecs) == 0:
        fail_count += 1
        print(dp)
print(fail_count)
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
            temp.append(path2file[path])
    with open('data/'+name+'.pkl', 'wb') as f:
        pickle.dump(temp, f)

