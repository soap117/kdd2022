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
def find_location(file):
    anno = file['anno']
    count = 0
    for content in file['file']['contents']:
        for tooltip in content['tooltips']:
            if tooltip['translation'] == anno and tooltip['origin'] == file['original_key']:
                file['sentence'] = content['text']
                file['position'] = (tooltip['l'], tooltip['l']+len(tooltip['origin']))
                file['origin_key'] = tooltip['origin']
                count += 1
    return file
def creat_sentence(data_new):
    sentence_format = {}
    for file in data_new:
        anno = file['file']
        src = anno['src']
        src = re.sub('\*\*', '', src)
        src = src.replace('\n', '').replace('。。', '。')
        tar = anno['tar']
        tar = re.sub('\*\*', '', tar)
        tar = tar.replace('\n', '').replace('。。', '。')
        if src[-1] == '。' and tar[-1] != '。':
            tar += '。'
        if tar[-1] == '。' and src[-1] != '。':
            src += '。'
        data_key = None
        src_sts = src.split('。')
        tar_sts = tar.split('。')
        dt = len(tar_sts) - len(src_sts)
        if dt == 0:
            for src_st, tar_st in zip(src_sts, tar_sts):
                if file['original_key'] in src_st and src_st != tar_st:
                    file['src_st'] = src_st
                    file['tar_st'] = tar_st
                    pos = re.search(file['original_key'], src_st)
                    file['position'] = pos.regs
                    data_key = {'key': file['key'], 'origin':file['origin_key'], 'anno': file['anno'], 'urls': file['urls'], 'rsecs': file['rsecs'],
                                'rpsecs': file['rpsecs'], 'pos': file['position']}
                    break
        if data_key is not None:
            file_sen = file['file']['textid'] + file['src_st']
            if file_sen in sentence_format:
                sentence_format[file_sen]['data'].append(data_key)
            else:
                sentence_format[file_sen] = {}
                sentence_format[file_sen]['data'] = [data_key]
                sentence_format[file_sen]['src_st'] = file['src_st']
                sentence_format[file_sen]['tar_st'] = file['tar_st']
                sentence_format[file_sen]['textid'] = file['file']['textid']
    return list(sentence_format.values())

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
for fid, (file, index_list) in tqdm(enumerate(zip(my_data[0:1000], group_list[0:1000]))):
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
        if max(batch_scores) <= 2 and key not in top_n_sentence and can_simi*inversed_punishment < 0.6:
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
    file = find_location(file)
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
sentence_format = creat_sentence(data_new)
with open('mydata_sen_clean_v4_sec_sub_trn.pkl', 'wb') as f:
    pickle.dump(sentence_format, f)
print(count_f)
#with open('mydata_new_clean_v4_sec_sub_trn.pkl', 'wb') as f:
#    pickle.dump(data_new, f)