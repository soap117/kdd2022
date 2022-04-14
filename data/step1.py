import os
import pickle
import re

if os.path.exists('mydata_new_clean.pkl'):
    with open('mydata_done_baidu.pkl', 'rb') as f:
        mark_done = pickle.load(f)
    with open('mydata_new_clean.pkl', 'rb') as f:
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
fail_count = 0
new_data = []
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
            if len(rpsec) == 0:
                if url not in url2secs or len(url2secs[url])==0:
                    fail_count += 1
                    continue
                rpsec = url2secs[url]
            title = rpsec[-1]
            if not isChinese(title):
                print(title)
                fail_count += 1
                continue
            if url not in url2secs:
                url2secs[url] = rpsec
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
    new_data.append(dp)
print(fail_count)
with open('mydata_new_clean_v2.pkl', 'wb') as f:
    pickle.dump(new_data, f)
