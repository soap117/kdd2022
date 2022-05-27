import pickle
import torch
import numpy as np
from tqdm import tqdm
import re
from transformers import BertTokenizer, BertModel
from keras.preprocessing.sequence import pad_sequences
import os
import asyncio
from pprint import pprint
import argparse
import sys
sys.path.append("../../")
from utils.kmp import KMP
import numpy as np
from align import deal_one, deal_anno
import json
kmp = KMP()

CLS = '[CLS]'
SEP = '[SEP]'
bert_model = 'hfl/chinese-bert-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(bert_model)

def preprocess(dataset, path_to):
    src_all, src_ids, src_tokens = [], [], []
    tar_masks = []
    keywordset_list = []

    for data in tqdm(dataset, ascii=True, ncols=50):
        contents = data['contents']
        tokens = [CLS]
        masks = [0]
        _keywords = []

        for content in contents:
            src = deal_anno(content['text'])
            _src_tokens = tokenizer.tokenize(re.sub('\*\*', '', src).lower())
            src_masks = np.array([0 for _ in range(len(_src_tokens))])
            for tooltip in content['tooltips']:
                key = tooltip['origin']
                key = deal_anno(key)
                keyword_tokens = tokenizer.tokenize(key)
                i = kmp.kmp(_src_tokens, keyword_tokens)
                l = len(keyword_tokens)
                if i != -1:
                    src_masks[i] = 1
                    if l >= 2:
                        # print("get keyword token len >= 2")
                        src_masks[i + l - 1] = 3
                        src_masks[i + 1:i + l - 1] = 3
                    _keywords.append(key)
            tokens += _src_tokens
            masks += list(src_masks)

        tokens += [SEP]
        masks += [0]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        src_ids.append(ids)
        tar_masks.append(masks)
        keywordset_list.append(_keywords)
        src_tokens.append(tokens)

    print(len(src_ids))
    src_ids_smaller, tar_masks_smaller,keywords_smaller,src_tokens_smaller = [], [], [], []
    max_len = 512
    indexs = []
    for i,(src, masks,keywords, tokens) in enumerate(zip(src_ids, tar_masks,keywordset_list, src_tokens)):
        if len(src) < max_len and len(src) > 2:
            src_ids_smaller.append(src)
            tar_masks_smaller.append(masks)
            keywords_smaller.append(keywords)
            indexs.append(i)
            src_tokens_smaller.append(tokens)

    src_ids, tar_masks, keywordset_list, src_tokens = src_ids_smaller, tar_masks_smaller, keywords_smaller,src_tokens_smaller
    print(len(src_ids))

    tag_values = [0, 1, 2]
    tag2idx = {t: i for i, t in enumerate(tag_values)}

    src_ids = pad_sequences(src_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    tar_masks = pad_sequences(tar_masks, maxlen=max_len, dtype="long", value=tag2idx[2], truncating="post", padding="post")

    src_masks = [[float(i != 0.0) for i in ii] for ii in src_ids]

    with open(os.path.join(path_to, 'src_ids.pkl'), 'wb') as f:
        pickle.dump(src_ids, f)
    with open(os.path.join(path_to, 'src_masks.pkl'), 'wb') as f:
        pickle.dump(src_masks, f)
    with open(os.path.join(path_to, 'tar_masks.pkl'), 'wb') as f:
        pickle.dump(tar_masks, f)
    with open(os.path.join(path_to, 'keywordset_list.pkl'), 'wb') as f:
        pickle.dump(keywordset_list, f)

    with open(os.path.join(path_to,'data.txt'),'w', encoding='utf-8') as f:
        for src, masks in zip(src_tokens, tar_masks):
            for token, mask in zip(src, masks):
                f.write(token+' '+str(mask)+'\n')
            f.write('\n')


def main():
    dataset = json.load(open('../data/dataset_new_3.json', 'r', encoding='utf-8'))
    total = len(dataset)
    print('train dataset:')
    preprocess(dataset[:int(total/10*8)], './data/train')
    print('test dataset:')
    preprocess(dataset[int(total/10*8):int(total/10*9)], './data/test')
    print('valid dataset:')
    preprocess(dataset[int(total/10*9):], './data/valid')
    print('done')

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', type=int)
    # args = parser.parse_args()
    # print(args.i)
    # main(args.i)
    main()



