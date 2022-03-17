import os
import pickle
import time

count ={}
my_data = []
url2secs = {}
mark_done = []
url_done = set()
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
import justext

lock = threading.Lock()
lock_m = threading.Lock()
lock_d = threading.Lock()

class myThread(threading.Thread):
    def __init__(self, name, files):
        threading.Thread.__init__(self)
        self.name = name
        self.files = files
        stops = open('stop_words.txt', 'r', encoding='utf-8').readlines()
        self.stops = tuple(stops)

    def web_read(self, url):
        if 'baike.baidu' not in url or len(url) < 2:
            #print('Skipping')
            return [], True
        lock_d.acquire()
        if url in url_done:
            lock_d.release()
            return [], False
        lock_d.release()
        time.sleep(0.5*np.random.rand()+0.5)
        #print(url)
        flag = False
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
            print('Failed to access %s' % url)
            return [], True
        # r.encoding = r.apparent_encoding
        html = r.text
        if r.status_code != 200:
            print('Failed to access %s' % url)
            return [], True
        if 'baidu' in url:
            secs = re.findall('<div class="para" label-module="para">(.*)</div>', html)
        else:
            secs_ = justext.justext(r.content, stoplist=self.stops)
            secs = []
            for sec_ in secs_:
                if sec_.cf_class != 'short' and len(sec_.text) > 15:
                    secs.append(sec_.text)

        title = re.findall('<title>(.*)</title>', html)[0]
        secs.append(title)
        for sid, s in enumerate(secs):
            new_s = re.sub('<[^<>]*>', '', s)
            secs[sid] = new_s
        if len(secs) == 0:
            secs_ = justext.justext(r.content, stoplist=self.stops)
            secs = []
            for sec_ in secs_:
                if sec_.cf_class != 'short' and len(sec_.text) > 15:
                    secs.append(sec_.text)
        lock.acquire()
        url2secs[url] = secs
        lock.release()
        return secs, flag

    def run(self):
        global my_data, mark
        print('start thread %s' %self.name)
        print(len(self.files))
        for fid, file in enumerate(self.files):
            # print(file)
            data = file
            for sec_data in data['contents']:
                for ann_data in sec_data['tooltips']:
                    key = ann_data['origin']
                    anno = ann_data['translation']
                    urls = []
                    rsecs = []
                    rpo_secs = []
                    for ref_data in ann_data['sources']:
                        url = ref_data['link']
                        rsec = ref_data['reference']
                        try:
                            page_secs, flag = self.web_read(url)
                        except:
                            flag = True
                            page_secs = []
                        if not flag:
                            urls.append(url)
                            rsecs.append(rsec)
                            rpo_secs.append(page_secs)
                            lock_d.acquire()
                            url_done.add(url)
                            lock_d.release()
                    if not flag:
                        field = {'file': file, 'key': key, 'anno': anno, 'urls': urls, 'rsecs': rsecs, 'rpsecs': rpo_secs, }
                        lock.acquire()
                        my_data.append(field)
                        mark_done.append(file)
                        if len(mark_done) % 100 == 0:
                            print('Saving middle result finished=%d' %len(mark_done))
                            with open('mydata_done_baidu.pkl', 'wb') as f:
                                pickle.dump(mark_done, f)
                            with open('mydata_new_baidu_.pkl', 'wb') as f:
                                pickle.dump(my_data, f)
                            with open('./mydata_url2secs_new_baidu.pkl', 'wb') as f:
                                pickle.dump(url2secs, f)
                            with open('./mydata_url_new_baidu.pkl', 'wb') as f:
                                pickle.dump(url_done, f)
                        print(len(mark_done))
                        lock.release()
'''
if os.path.exists('mydata_done_baidu.pkl'):
    with open('mydata_done_baidu.pkl', 'rb') as f:
        mark_done = pickle.load(f)
    with open('mydata_new_baidu_.pkl', 'rb') as f:
        my_data = pickle.load(f)
    with open('./mydata_url2secs_new_baidu.pkl', 'rb') as f:
        url2secs = pickle.load(f)
    with open('./mydata_url_new_baidu.pkl', 'rb') as f:
        url_done = pickle.load(f)
'''
raw_data_new = json.load(open('./dataset_new_2.json', 'r', encoding='utf-8'))
file_list = []

mark_done_set = set(mark_done)
for one in raw_data_new:
    if one['textid'] not in mark_done_set:
        file_list.append(one)

len_file = len(file_list)
ind = len_file//4 + 1
thread_list = [1, 2, 3, 4]
threads = []
for temp in thread_list:
    thread = myThread(str(temp), file_list[(temp-1)*ind:temp*ind])
    thread.start()
    threads.append(thread)
for t in threads:
    t.join()
print(len(my_data))
with open('mydata_done_baidu.pkl', 'wb') as f:
    pickle.dump(mark_done, f)
with open('mydata_new_baidu_.pkl', 'wb') as f:
    pickle.dump(my_data, f)
with open('./mydata_url2secs_new_baidu.pkl', 'wb') as f:
    pickle.dump(url2secs, f)
with open('./mydata_url_new_baidu.pkl', 'wb') as f:
    pickle.dump(url_done, f)


