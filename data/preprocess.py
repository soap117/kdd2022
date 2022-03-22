import os
import pickle
import re

if os.path.exists('mydata_done_baidu.pkl'):
    with open('mydata_done_baidu.pkl', 'rb') as f:
        mark_done = pickle.load(f)
    with open('mydata_new_baidu_.pkl', 'rb') as f:
        my_data = pickle.load(f)
    with open('./mydata_url2secs_new_baidu.pkl', 'rb') as f:
        url2secs = pickle.load(f)
    with open('./mydata_url_new_baidu.pkl', 'rb') as f:
        url_done = pickle.load(f)
path2file = {}
def isChinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff' or ch == ' ' or ch == '_' or ch == '-':
            return True
    return False

for dp in my_data:
    new_urls = []
    new_rsecs = []
    new_rpsecs = []
    url_set = set()
    for url, rsec, rpsec in zip(dp['urls'], dp['rsecs'], dp['rpsecs']):
        if url not in url_set:
            if len(rpsec) == 0:
                rpsec = url2secs[url]
            title = rpsec[-1]
            if not isChinese(title):
                print(title)

            new_urls.append(url)
            new_rsecs.append(rsec)
            new_rpsecs.append(rpsec)
            url_set.add(url)
    dp['urls'] = new_urls
    dp['rsecs'] = new_rsecs
    dp['rpsecs'] = new_rpsecs
    path2file[dp['file']['textid']] = dp
    if len(new_rpsecs) == 0:
        print(dp)
with open('mydata_new_clean.pkl', 'wb') as f:
    pickle.dump(my_data, f)
names = ['train', 'valid', 'test']
sets = {}
for name in names:
    sets[name] = []
    with open(name+'/dataset.pkl', 'rb') as f:
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
    with open('./'+name+'.pkl', 'wb') as f:
        pickle.dump(temp, f)

