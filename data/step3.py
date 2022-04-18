import os
import pickle
import time


from tqdm import tqdm
import requests
import time
import numpy as np
import queue
import threading
from lxml import etree              # 导入库
from bs4 import BeautifulSoup
import re
import json
def is_english(key):
    return key.encode('UTF-8').isalpha()
def return_eng(key):
    for sub_key in key.split(' '):
        if is_english(sub_key):
            return sub_key
    return None
lock = threading.Lock()
lock_m = threading.Lock()
lock_d = threading.Lock()
glob_time = time.time()
apiKey = 'HU8af50ccf0318014312fR0R'
mark_done = {}

g_count = 0
class myThread(threading.Thread):
    def __init__(self, name, files):
        threading.Thread.__init__(self)
        self.name = name
        self.files = files

    def run(self):
        global my_data, mark_done, g_count, proxies, glob_time
        print('start thread %s' %self.name)
        print(len(self.files))
        for fid, file in enumerate(self.files):
            # print(file)
            key = return_eng(file['key'])
            if key is None:
                continue
            key = key.replace(' ', '%20').lower()
            if key in mark_done:
                file['key'] = mark_done[key]
                continue
            time.sleep(0.5)
            url = 'http://www.mcd8.com/w/%s' %key
            headers = {
                'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
            }
            try_count = 0
            while try_count < 3:
                try:
                    r = requests.get(url, headers=headers, timeout=5)
                    break
                except Exception as e:
                    try_count += 1
                    print("trying %d time" % try_count)
                    wait_gap = 3
                    time.sleep((try_count + np.random.rand()) * wait_gap)
            if try_count >= 3:
                lock_m.acquire()
                time_now = time.time()
                lock_m.release()
                continue
            results = re.findall('<div class="content">(.+)</div>', r.text)
            name_set = set()
            for one in results:
                answers = re.sub('[^\u4e00-\u9fa5, ]', '', re.sub('<[^<>]*>',  ' ', one)).split('  ')
                flag = False
                for ans in answers:
                    if '医学' in ans:
                        flag = True
                if not flag:
                    continue
                for ans in answers:
                    if len(ans) > 0 and '词典' not in ans and '词条' not in ans:
                        name_set.add(ans.replace(' ', ''))
            for sub_key in file['key'].split(' '):
                name_set.add(sub_key.replace(' ', ''))
            name_list = list(name_set)
            new_key = ' '.join(name_list)
            if len(new_key) == 0:
                new_key = key
            lock_m.acquire()
            file['key'] = new_key
            mark_done[key] = new_key
            if key!=new_key:
                print("%s->%s" %(key, new_key))
            g_count += 1
            if g_count % 10 == 0:
                print(g_count)
            if g_count%100 == 0:
                with open('mydata_new_clean_v4.pkl', 'wb') as f:
                    pickle.dump(my_data, f)

            lock_m.release()
if os.path.exists('mydata_new_clean_v4.pkl'):
    with open('mydata_new_clean_v3.pkl', 'rb') as f:
        my_data = pickle.load(f)
else:
    with open('mydata_new_clean_v3.pkl', 'rb') as f:
        my_data = pickle.load(f)

len_file = len(my_data)
ind = len_file//4 + 1
thread_list = [1, 2, 3, 4]
threads = []
for temp in thread_list:
    thread = myThread(str(temp), my_data[(temp-1)*ind:temp*ind])
    thread.start()
    threads.append(thread)
for t in threads:
    t.join()
print(len(my_data))
with open('mydata_new_clean_v4.pkl', 'wb') as f:
    pickle.dump(my_data, f)


