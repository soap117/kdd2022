
import pickle
import torch
from tqdm import tqdm
import re
import random
import os
import json
import numpy as np
from align import creat_sentence, deal_one, deal_anno
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
def aligneddata(dataset,path):
    count = 0
    # with open(os.path.join(path,'dataset.pkl'),'rb') as f:
    #     dataset = pickle.load(f)
    # src_all = [u[0] for u in dataset]
    # tar_all = [u[1] for u in dataset]
    src_all = [u['src'] for u in dataset]
    tar_all = [u['tar'] for u in dataset]
    contents_all = [u['contents'] for u in dataset]

    for i in range(len(src_all)):
        src = src_all[i]
        src = re.sub('\*\*', '', src).lower()
        src = src.replace('\n', '').replace('。。', '。')
        src_all[i] = src

    for i in range(len(tar_all)):
        tar = tar_all[i]
        tar = re.sub('\*\*', '', tar).lower()
        tar = tar.replace('\n', '').replace('。。', '。')
        tar_all[i] = tar

    print(len(dataset))

    dataset_new = []
    dataset_new_para = []
    dataset = []
    src_ids = []
    tar_masks = []

    for src, tar, content in tqdm(zip(src_all, tar_all, contents_all)):
        if len(src)==0 or len(tar)==0: continue
        # if src[-1] == '。' and tar[-1] != '。':
        #     tar += '。'
        # if tar[-1] == '。' and src[-1] != '。':
        #     src += '。'
        # srcs = src.strip('。').split('。')
        # tars = tar.strip('。').split('。')
        src_mark = ''
        srcs, tars = creat_sentence(src, tar)
        tars_sec = '。'.join(tars)
        srcs_sec = '。'.join(srcs)
        #src_tokens = ['[CLS]'] + tokenizer.tokenize(srcs_sec) + ['[SEP]']
        #masks = np.array([0 for _ in range(len(src_tokens))])
        src_tokens = []
        masks = []
        for sen in content:
            sen_text = sen['text']
            sen_text = deal_one(sen_text)
            sen_tokens = tokenizer.tokenize(sen_text)
            sen_masks = np.array([0 for _ in range(len(sen_tokens))])
            for raw_anno in sen['tooltips']:
                anno_name = deal_anno(raw_anno['origin'])
                find_rs = None
                gap = ''
                while find_rs is None and len(gap)<3:
                    find_rs = re.search(anno_name+gap+'（[^（）]*）', tars_sec)
                    gap += '.'
                if find_rs is None:
                    continue
                comp_anno_words = tars_sec[find_rs.regs[0][0]:find_rs.regs[0][1]]
                if comp_anno_words not in srcs_sec:
                    is_real_anno = True
                else:
                    is_real_anno = False
                if is_real_anno:
                    anno_tokens = tokenizer.tokenize(anno_name)
                    for ith in range(len(sen_tokens)):
                        if sen_tokens[ith:ith+len(anno_tokens)] == anno_tokens:
                            sen_masks[ith+len(anno_tokens)-1] = 3
                            sen_masks[ith+1:ith+len(anno_tokens)-1] = 2
                            sen_masks[ith] = 1
                            break
                    count += 1
            src_tokens += sen_tokens
            masks.append(sen_masks)
        src_tokens = ['[CLS]'] + src_tokens + ['[SEP]']
        masks = [np.zeros(1)] + masks + [np.zeros(1)]
        masks = np.concatenate(masks, axis=0)
        ids = tokenizer.convert_tokens_to_ids(src_tokens)
        src_ids.append(ids)
        tar_masks.append(masks)

    print(count)
    with open(os.path.join(path, 'src_ids.pkl'), 'wb') as f:
        pickle.dump(src_ids, f)
    with open(os.path.join(path, 'tar_masks.pkl'), 'wb') as f:
        pickle.dump(tar_masks, f)


def main():
    dataset = json.load(open('../data/dataset_new_3.json', 'r', encoding='utf-8'))
    total = len(dataset)
    print('train dataset:')
    aligneddata(dataset[:int(total/10*8)],'./data/train')
    print('test dataset:')
    aligneddata(dataset[int(total/10*8):int(total/10*9)],'./data/test')
    print('valid dataset:')
    aligneddata(dataset[int(total/10*9):],'./data/valid')


if __name__ == '__main__':
    main()