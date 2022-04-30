import os.path
import pickle
import torch
from tqdm import tqdm
import re
import random
import os

def aligneddata(path):
    with open(os.path.join(path,'dataset.pkl'),'rb') as f:
        dataset = pickle.load(f)
    src_all = [u[0] for u in dataset]
    tar_all = [u[1] for u in dataset]

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
    dts = {}
    for src, tar in zip(src_all, tar_all):
        if len(src)==0 or len(tar)==0: continue
        if src[-1] == '。' and tar[-1] != '。':
            tar += '。'
        if tar[-1] == '。' and src[-1] != '。':
            src += '。'
        src_sts = src.split('。')
        tar_sts = tar.split('。')
        dt = len(tar_sts) - len(src_sts)
        if dt == 0:
            dataset_new.append((src, tar))
        if dt in dts.keys():
            dts[dt] += 1
        else:
            dts[dt] = 1
    print(len(dataset_new))
    print(dts)
    with open(os.path.join(path,'dataset-aligned.pkl'), 'wb') as f:
        pickle.dump(dataset_new, f)


def main():
    print('train dataset:')
    aligneddata('./train')
    print('test dataset:')
    aligneddata('./test')
    print('valid dataset:')
    aligneddata('./valid')


if __name__ == '__main__':
    main()