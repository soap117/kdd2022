import os
import pickle
import time


from tqdm import tqdm
import requests
import time
import numpy as np
import jieba
import threading
from lxml import etree              # 导入库
from bs4 import BeautifulSoup
import re
import json
from rank_bm25 import BM25Okapi
def is_english(key):
    return key.encode('UTF-8').isalpha()
lock = threading.Lock()
lock_m = threading.Lock()
lock_d = threading.Lock()
glob_time = time.time()
mark_down = {}
if os.path.exists('mydata_new_clean_v4.pkl'):
    with open('mydata_new_clean_v4.pkl', 'rb') as f:
        my_data = pickle.load(f)
else:
    with open('mydata_new_clean_v4.pkl', 'rb') as f:
        my_data = pickle.load(f)
data_new = []
for fid, file in tqdm(enumerate(my_data)):
    # print(file)
    key = file['key']
    anno = file['anno']
    key_anno = key + ' ' +anno
    sentences = ('。'.join(file['rsecs'])).split('。')
    sentences = [x for x in sentences if len(x) > 3]
    tokenized_corpus = [jieba.lcut(doc) for doc in sentences]
    if len(tokenized_corpus) == 0 or len(tokenized_corpus[0]) == 0:
        continue
    bm25_sentences = BM25Okapi(tokenized_corpus)
    key_cut = jieba.lcut(anno)
    top_n_sentences = bm25_sentences.get_top_n(key_cut, sentences, 3)
    file['rsecs'] = [top_n_sentences[0]]
    sentences = []
    for group in file['rpsecs']:
        sentences += group
    sentences = ('。'.join(sentences)).split('。')
    sentences = [x for x in sentences if len(x) > 3]
    tokenized_corpus = [jieba.lcut(doc) for doc in sentences]
    if len(tokenized_corpus) == 0 or len(tokenized_corpus[0]) == 0:
        continue
    bm25_sentences = BM25Okapi(tokenized_corpus)
    key_cut = jieba.lcut(anno)
    top_n_sentences = bm25_sentences.get_top_n(key_cut, sentences, 10)
    file['rpsecs'] = [top_n_sentences]
    data_new.append(file)
    #print(top_n_sentences)

with open('mydata_new_clean_v4_sec.pkl', 'wb') as f:
    pickle.dump(data_new, f)