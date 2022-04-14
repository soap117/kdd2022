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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth = SmoothingFunction()
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
    sen_set = set()
    id_list = []
    if len(''.join(file['rsecs']))==0 and len(file['rpsecs'])>0:
        file['rsecs'] = file['rpsecs'][0]
    for example in file['rsecs']:
        exp_id_list = []
        sentences = example.split('。')
        for sentence in sentences:
            if len(sentence) <= 3:
                continue
            if sentence in sen_set:
                continue
            else:
                sen_set.add(sentence)
            if sentence in sen2id.keys():
                exp_id_list.append(sen2id[sentence])
            else:
                sen2id[sentence] = len(sen2id)
                exp_id_list.append(sen2id[sentence])
        id_list.append(exp_id_list)
    group_list.append(id_list)
complete_corpus = []
for sen in tqdm(sen2id.keys()):
    tokenized_corpus = jieba.lcut(sen)
    complete_corpus.append(tokenized_corpus)
bm25_sentences = BM25Okapi(complete_corpus)
for fid, (file, index_list) in tqdm(enumerate(zip(my_data, group_list))):
    if len(index_list) <= 0:
        continue
    # print(file)
    key = file['key']
    anno = file['anno']
    key_anno = key + ' ' +anno
    key_cut = jieba.lcut(anno)
    flag = True
    top_n_sentences = []
    for exp_id_list in index_list:
        if len(exp_id_list) == 0:
            continue
        batch_scores = bm25_sentences.get_batch_scores(key_cut, exp_id_list)
        top_n_sentence = ''.join(complete_corpus[exp_id_list[np.argmax(batch_scores)]])
        can_simi = sentence_bleu([complete_corpus[exp_id_list[np.argmax(batch_scores)]]], key_cut, weights=(0.5, 0.5),
                                 smoothing_function=smooth.method1)
        inversed_punishment = 1 / np.exp(1 - len(complete_corpus[exp_id_list[np.argmax(batch_scores)]]) / len(key_cut))
        if max(batch_scores) <= 2 and key not in top_n_sentence and can_simi*inversed_punishment < 0.75:
            if anno in key:
                key = anno
                file['key'] = key
                flag = False
        else:
            flag = False
        top_n_sentences.append(top_n_sentence)
    if flag:
        count_f += 1
        print(key)
        print(anno)
        print(top_n_sentences)
        print('--------------')
        continue
    file['rsecs'] = top_n_sentences
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