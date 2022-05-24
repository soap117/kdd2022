import os.path
import pickle
import torch
from tqdm import tqdm
import re
import random
import os
import json
from align import creat_sentence

def aligneddata(dataset,path):
    # with open(os.path.join(path,'dataset.pkl'),'rb') as f:
    #     dataset = pickle.load(f)
    # src_all = [u[0] for u in dataset]
    # tar_all = [u[1] for u in dataset]
    src_all = [u['src'] for u in dataset]
    tar_all = [u['tar'] for u in dataset]

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

    for src, tar in zip(src_all, tar_all):
        if len(src)==0 or len(tar)==0: continue
        # if src[-1] == '。' and tar[-1] != '。':
        #     tar += '。'
        # if tar[-1] == '。' and src[-1] != '。':
        #     src += '。'
        # srcs = src.strip('。').split('。')
        # tars = tar.strip('。').split('。')
        srcs, tars = creat_sentence(src, tar)
        if len(srcs) == len(tars):
            dataset.append((src,tar))
            for u,v in zip(srcs, tars):
                if len(u)>2 and len(v)>2:
                    dataset_new.append((u, v))
        tars_sec = '。'.join(tars)
        srcs_sec = '。'.join(srcs)
        dataset_new_para.append((srcs_sec, tars_sec))

    print(len(dataset))
    with open(os.path.join(path, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
    print(len(dataset_new))
    with open(os.path.join(path,'dataset-aligned.pkl'), 'wb') as f:
        pickle.dump(dataset_new, f)
    with open(os.path.join(path,'dataset-aligned-para.pkl'), 'wb') as f:
        pickle.dump(dataset_new_para, f)


def main():
    dataset = json.load(open('./dataset_new_3.json', 'r', encoding='utf-8'))
    total = len(dataset)
    print('train dataset:')
    aligneddata(dataset[:int(total/10*8)],'./train')
    print('test dataset:')
    aligneddata(dataset[int(total/10*8):int(total/10*9)],'./test')
    print('valid dataset:')
    aligneddata(dataset[int(total/10*9):],'./valid')


if __name__ == '__main__':
    main()