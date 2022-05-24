import pickle
import torch
import numpy as np
from tqdm import tqdm
import re
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import os
from pprint import pprint

CLS = '[CLS]'
SEP = '[SEP]'

def tokenize(tokenizer, txt):
    src = re.sub('\*\*', '', txt).lower()
    tokens = tokenizer.tokenize(src)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return ids

def preprocess_sec(path_from):
    with open(os.path.join(path_from,'dataset-aligned.pkl'), 'rb') as f:
        dataset_aligned = pickle.load(f)
    src_all, tar_all, ref_all, src_ids, tar_ids = [], [], [], [], []
    bert_model = 'hfl/chinese-bert-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    srcs = []
    tars = []
    for data in dataset_aligned:
        content = data[2]
        src = data[0].strip()
        tar = data[1].strip()
        src_list = src.split('。')
        tar_list = tar.split('。')

        if src_list[-1] == '' and len(content)!=len(src_list):
            src_list = src_list[:-1]
        if len(content)!=len(src_list):
            continue

        src = '。'.join(src_list)
        tar = '。'.join(tar_list)
        srcs.append(src)
        tars.append(tar)
    return srcs, tars




