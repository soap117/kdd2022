import pickle
import torch
import numpy as np
from tqdm import tqdm
import re
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import os
from pprint import pprint


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
keywords_all.pop('')


def main(path_to, indexs):
    src_ids, src_masks, tar_masks, keywordset_list = [],[],[],[]

    for index in indexs:
        with open(os.path.join(path_to, 'src_ids_{}.pkl'.format(index)), 'rb') as f:
            _src_ids = pickle.load(f)
        with open(os.path.join(path_to, 'src_masks_{}.pkl'.format(index)), 'rb') as f:
            _src_masks = pickle.load(f)
        with open(os.path.join(path_to, 'tar_masks_{}.pkl'.format(index)), 'rb') as f:
            _tar_masks = pickle.load(f)
        with open(os.path.join(path_to, 'keywordset_list_{}.pkl'.format(index)), 'rb') as f:
            _keywordset_list = pickle.load(f)
        src_ids.extend(_src_ids)
        src_masks.extend(_src_masks)
        tar_masks.extend(_tar_masks)
        keywordset_list.extend(_keywordset_list)

    for i, (src_id, src_mask, tar_mask, keywordset_list) in tqdm(enumerate(zip(src_ids, src_masks, tar_masks, keywordset_list))):
        for l in range(2, 6):
            for k in range(len(src_id)):
                word = ''.join(tokenizer.convert_ids_to_tokens(src_id[k:k+l]))
                if word in keywords_all.keys():
                    word_mask = [1 if _==0 else 3 for _ in range(l)]
                    if len(word_mask)>1: word_mask[-1]=4
                    tar_mask[k:k+l] = word_mask

        tar_masks[i] = tar_mask


    with open(os.path.join(path_to, 'src_ids.pkl'), 'wb') as f:
        pickle.dump(src_ids, f)
    with open(os.path.join(path_to, 'src_masks.pkl'), 'wb') as f:
        pickle.dump(src_masks, f)
    with open(os.path.join(path_to, 'tar_masks.pkl'), 'wb') as f:
        pickle.dump(tar_masks, f)
    with open(os.path.join(path_to, 'keywordset_list.pkl'), 'wb') as f:
        pickle.dump(keywordset_list, f)
    print(len(src_ids))

    with open(os.path.join(path_to,'data.txt'),'w', encoding='utf-8') as f:
        for src, masks in zip(src_ids, tar_masks):
            for id, mask in zip(src, masks):
                f.write(tokenizer.convert_ids_to_tokens([id])[0]+' '+str(mask)+'\n')
            f.write('\n')

if __name__ == '__main__':
    main('./data/train', [1682,3365,5048,6732,8415,10098,11781,13465])
    main('./data/test', [209,419,629,839,1049,1259,1469,1679])
    main('./data/valid', [209,419,629,839,1049,1259,1469,1680])




