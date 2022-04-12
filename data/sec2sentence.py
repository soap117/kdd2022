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
data_new = []
count_f = 0
complete_corpus = []
group_list = []
sen2id = {}
for fid, file in tqdm(enumerate(my_data)):
    sentences = ('。'.join(file['rsecs'])).split('。')
    id_list = []
    sen_set = set()
    for sentence in sentences:
        if len(sentence) <= 3:
            continue
        if sentence in sen_set:
            continue
        else:
            sen_set.add(sentence)
        if sentence in sen2id.keys():
            id_list.append(sen2id[sentence])
        else:
            sen2id[sentence] = len(sen2id)
            id_list.append(sen2id[sentence])
    group_list.append(id_list)
complete_corpus = []
for sen in tqdm(sen2id.keys()):
    tokenized_corpus = jieba.lcut(sen)
    complete_corpus.append(tokenized_corpus)
bm25_sentences = BM25Okapi(complete_corpus)
for fid, (file, index_list) in tqdm(enumerate(zip(my_data, group_list))):
    if len(index_list) <=0:
        continue
    # print(file)
    key = file['key']
    anno = file['anno']
    key_anno = key + ' ' +anno
    key_cut = jieba.lcut(anno)
    batch_scores = bm25_sentences.get_batch_scores(key_cut, index_list)
    top_n_sentence = ''.join(complete_corpus[index_list[np.argmax(batch_scores)]])
    #print(anno)
    #print(top_n_sentence)
    if max(batch_scores) <= 2 and key not in top_n_sentence:
        if anno in key:
            key = anno
            file['key'] = key
        else:
            count_f += 1
            print(key)
            print(anno)
            print(top_n_sentence)
            print('--------------')
            continue
    file['rsecs'] = [top_n_sentence]
    sentences = []
    for gid, group in enumerate(file['rpsecs']):
        sentences = group[0:-1]
        title = group[-1]
        sentences = ('。'.join(sentences)).split('。')
        sentences = [x for x in sentences if len(x) > 3]
        tokenized_corpus = [jieba.lcut(doc) for doc in sentences]
        if len(tokenized_corpus) == 0 or len(tokenized_corpus[0]) == 0:
            continue
        bm25_sentences_local = BM25Okapi(tokenized_corpus)
        key_cut = jieba.lcut(anno)
        top_n_sentences = bm25_sentences_local.get_top_n(key_cut, sentences, 5)
        top_n_sentences.append(title)
        file['rpsecs'][gid] = top_n_sentences
    if len(file['rpsecs']) == 0:
        print('here')
    data_new.append(file)
    #print(top_n_sentences)
print(count_f)
with open('mydata_new_clean_v4_sec.pkl', 'wb') as f:
    pickle.dump(data_new, f)