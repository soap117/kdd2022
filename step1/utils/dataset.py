PAD, CLS = '[PAD]', '[CLS]'
SEP = '[SEP]'
from tqdm import tqdm
import re
import nltk
import pickle
import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
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
            while r<len(mask_tar) and mask_tar[r] != 0:
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



def build_dataset(src_ids_path, tar_masks_path):
    token_ids_srcs = pickle.load(open(src_ids_path, 'rb'))
    if isinstance(token_ids_srcs, dict):
        token_ids_srcs = token_ids_srcs['pad']
    mask_tars = pickle.load(open(tar_masks_path,'rb'))
    mask_tars = np.array(mask_tars)
    dataset = []
    for token_ids_src, mask_tar in zip(token_ids_srcs, mask_tars):
        indicator = obtain_indicator(token_ids_src, mask_tar)
        dataset.append((token_ids_src, mask_tar, indicator))
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
        src_ids = [_[0] for _ in datas]
        max_len = min(np.max([len(u) for u in src_ids]), 512)
        tar_ids = [_[1] for _ in datas]
        indicator = [_[2] for _ in datas]
        src_ids = torch.LongTensor(pad_sequences(src_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post"))
        tar_ids = torch.LongTensor(pad_sequences(tar_ids, maxlen=max_len, dtype="long", value=4, truncating="post", padding="post"))
        indicator = torch.LongTensor(
            pad_sequences(indicator, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post"))
        src_masks = torch.ones_like(src_ids)
        src_masks[src_ids == 0] = 0

        return (src_ids.to(self.device), src_masks.to(self.device), indicator.to(self.device)), tar_ids.to(self.device)

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


