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

lock = threading.Lock()
lock_m = threading.Lock()
lock_d = threading.Lock()
glob_time = time.time()
apiKey = 'HU8af50ccf0318014312fR0R'
mark_done = {}
def add_white_list(ip):
    url = "https://h.shanchendaili.com/api.html?action=addWhiteList&appKey={}&ip={}".format(apiKey, ip)
    res = requests.get(url)
    print(res.status_code)
    print(res.text)

def get_proxy():
    url = 'https://h.shanchendaili.com/api.html?action=get_ip&key={}&time=10&count=1&protocol=http&type=json&only=0'.format(apiKey)
    res = requests.get(url)
    print(res.status_code)
    data = json.loads(res.text)['list']
    ip = data[0]['sever']
    port = data[0]['port']
    return '{}:{}'.format(ip, port)


ip = '98.235.88.171'
add_white_list(ip)

g_count = 0
class myThread(threading.Thread):
    def __init__(self, name, files):
        threading.Thread.__init__(self)
        self.name = name
        self.files = files
        stops = open('stop_words.txt', 'r', encoding='utf-8').readlines()
        self.stops = tuple(stops)

    def run(self):
        global my_data, mark_done, g_count, proxies, glob_time
        print('start thread %s' %self.name)
        print(len(self.files))
        for fid, file in enumerate(self.files):
            # print(file)
            key = file['key']
            if key in mark_done:
                file['key'] = mark_done[key]
                continue
            time.sleep(0.1)
            url = 'https://api.ownthink.com/kg/ambiguous?mention=%s' %key
            headers = {
                'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
            }
            try_count = 0
            while try_count < 3:
                try:
                    r = requests.get(url, headers=headers, timeout=5, proxies=proxies)
                    break
                except Exception as e:
                    try_count += 1
                    print("trying %d time" % try_count)
                    wait_gap = 3
                    time.sleep((try_count + np.random.rand()) * wait_gap)
            if try_count >= 3:
                lock_m.acquire()
                time_now = time.time()
                if time_now-glob_time > 5:
                    flag = True
                    while flag:
                        try:
                            proxy = get_proxy()
                            proxies = {
                                "http": 'http://{}'.format(proxy),
                                "https": 'http://{}'.format(proxy),
                            }
                            glob_time = time_now
                            flag = False
                        except:
                            print('retry proxy')
                lock_m.release()
                continue
            rs = json.loads(r.text)
            rs = rs['data']
            name_set = set()
            for one in rs:
                name_set.add(re.sub(r'\[.*\]', '',one[0]))
            name_list = list(name_set)
            new_key = ' '.join(name_list)
            if len(new_key) == 0:
                new_key = key
            lock_m.acquire()
            file['old_key'] = file['key'].copy()
            file['key'] = new_key
            mark_done[key] = new_key
            if key!=new_key:
                print("%s->%s" %(key, new_key))
            g_count += 1
            if g_count % 10 == 0:
                print(g_count)
            if g_count%100 == 0:
                with open('mydata_new_clean_v3.pkl', 'wb') as f:
                    pickle.dump(my_data, f)

                with open('mydata_new_clean_v3_mark.pkl', 'wb') as f:
                    pickle.dump(mark_done, f)
            lock_m.release()
if os.path.exists('mydata_new_clean_v3.pkl'):
    with open('mydata_new_clean_v2.pkl', 'rb') as f:
        my_data = pickle.load(f)
    with open('mydata_new_clean_v3_mark.pkl', 'rb') as f:
        mark_done = pickle.load(f)
else:
    with open('mydata_new_clean_v2.pkl', 'rb') as f:
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
with open('mydata_new_clean_v3.pkl', 'wb') as f:
    pickle.dump(my_data, f)

with open('mydata_new_clean_v3_mark.pkl', 'wb') as f:
    pickle.dump(mark_done, f)

