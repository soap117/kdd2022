PAD, CLS = '[PAD]', '[CLS]'
SEP = '[SEP]'
from tqdm import tqdm
import re
import nltk
import pickle
import numpy as np
import torch

def try_match(checker, templete, token_ids):
    h = checker
    for ext in range(len(templete)):
        if templete[ext] != token_ids[h+ext]:
            return False
    return True

def obtain_indicator(token_ids_src, mask_tar):
    indicate = np.zeros_like(token_ids_src)
    for c_id in range(len(token_ids_src)):
        if mask_tar[c_id] == 1:
            l = c_id
            r = l+1
            while mask_tar[r] != 0:
                r += 1
            templete = token_ids_src[l:r]
            checker = r
            while checker<len(token_ids_src):
                if try_match(checker, templete, token_ids_src):
                    for k in range(checker, checker+len(templete)):
                        indicate[k] = 1
                    checker += len(templete)
                else:
                    checker += 1
    return indicate



def build_dataset(config, src_ids_path, src_masks_path, tar_masks_path):
    token_ids_srcs = pickle.load(open(src_ids_path, 'rb'))
    if isinstance(token_ids_srcs, dict):
        token_ids_srcs = token_ids_srcs['pad']
    mask_srcs = pickle.load(open(src_masks_path, 'rb'))
    mask_srcs = np.array(mask_srcs)
    mask_tars = pickle.load(open(tar_masks_path,'rb'))
    mask_tars = np.array(mask_tars)
    dataset = []
    for token_ids_src, mask_src, mask_tar in zip(token_ids_srcs, mask_srcs, mask_tars):
        indicator = obtain_indicator(token_ids_src, mask_tar)
        dataset.append((token_ids_src, mask_src, mask_tar, indicator))
    return dataset


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x_src = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        mask_src = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        mask_tar = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        indicator = torch.LongTensor([_[3] for _ in datas]).to(self.device)

        return (x_src, mask_src, indicator), mask_tar

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


