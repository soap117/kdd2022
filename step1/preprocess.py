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

CLS = '[CLS]'
SEP = '[SEP]'

bert_model = 'hfl/chinese-bert-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(bert_model)
# model = BertModel.from_pretrained(bert_model)
keywords_all = pickle.load(open('./data/keywords.pkl','rb'))
L = len(tokenizer)
print(L)
new_words = []
for keyword in tqdm(keywords_all.keys()):
    if keyword=='':
        print('empty!')
        continue
    tokenizer.add_tokens([keyword])
    if len(tokenizer)>L:
        L = len(tokenizer)
        new_words.append(keyword)

# model.resize_token_embeddings(len(tokenizer))
# for i, keyword in tqdm(enumerate(new_words)):
#     model.embeddings.word_embeddings.weight[-len(new_words)+i, :] = keywords_all[keyword]
print(len(tokenizer))
print(tokenizer.tokenize('白癜风得到的依从性关于 其他的痛风石'))


def preprocess(path_from, path_to, index):
    global keywords_all
    with open(os.path.join(path_from,'dataset-aligned.pkl'), 'rb') as f:
        dataset_aligned = pickle.load(f)
    src_all, src_ids = [], []
    tar_masks = []
    keywordset_list = []
    worker_num = 8

    dataset = []
    for i in range(worker_num):
        if i == index:
            for j, data in enumerate(
                    dataset_aligned[i * len(dataset_aligned) // worker_num: (i + 1) * len(dataset_aligned) // worker_num]):
                dataset.append((i * len(dataset_aligned) // worker_num + j, data))

    for _index, data in tqdm(dataset, ascii=True, ncols=50):
        content = data[2]
        src = data[0].strip()

        tokens = [CLS] + tokenizer.tokenize(re.sub('\*\*', '', src).lower()) + [SEP]
        masks = np.array([0 for _ in range(len(tokens))])
        _keywords = []

        for keyword in keywords_all.keys():
            if keyword not in src: continue
            keyword_tokens = tokenizer.tokenize(keyword)
            l = len(keyword_tokens)
            for i in range(len(tokens) - l):
                if tokens[i:i + l] == keyword_tokens:
                    masks[i:i + l] = 1
                    _keywords.append(keyword)

        ids = tokenizer.convert_tokens_to_ids(tokens)
        src_ids.append(ids)
        tar_masks.append(masks)
        keywordset_list.append(_keywords)


    print(len(src_ids))
    src_ids_smaller, tar_masks_smaller,keywords_smaller = [], [], []
    max_len = 512
    indexs = []
    for i,(src, masks,keywords) in enumerate(zip(src_ids, tar_masks,keywordset_list)):
        if len(src) < max_len and len(src) > 2:
            src_ids_smaller.append(src)
            tar_masks_smaller.append(masks)
            keywords_smaller.append(keywords)
            indexs.append(i)

    src_ids, tar_masks, keywordset_list = src_ids_smaller, tar_masks_smaller, keywords_smaller
    print(len(src_ids))

    tag_values = [0, 1, 2]
    tag2idx = {t: i for i, t in enumerate(tag_values)}

    src_ids = pad_sequences(src_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    tar_masks = pad_sequences(tar_masks, maxlen=max_len, dtype="long", value=tag2idx[2], truncating="post", padding="post")

    src_masks = [[float(i != 0.0) for i in ii] for ii in src_ids]

    with open(os.path.join(path_to, 'src_ids_{}.pkl'.format(index)), 'wb') as f:
        pickle.dump(src_ids, f)
    with open(os.path.join(path_to, 'src_masks_{}.pkl'.format(index)), 'wb') as f:
        pickle.dump(src_masks, f)
    with open(os.path.join(path_to, 'tar_masks_{}.pkl'.format(index)), 'wb') as f:
        pickle.dump(tar_masks, f)
    with open(os.path.join(path_to, 'keywordset_list_{}.pkl'.format(index)), 'wb') as f:
        pickle.dump(keywordset_list, f)

    # for ids, masks in zip(src_ids[:5], tar_masks[:5]):
    #     tokens = tokenizer.convert_ids_to_tokens(ids)
    #     for token, mask in zip(tokens, masks):
    #         print(token, mask)

def main(index):
    print('train dataset:')
    preprocess('../../data/train', './data/train',index)
    print('test dataset:')
    preprocess('../../data/test', './data/test',index)
    print('valid dataset:')
    preprocess('../../data/valid', './data/valid',index)
    print('done')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int)
    args = parser.parse_args()
    print(args.i)
    main(args.i)



