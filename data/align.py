import re, pickle, numpy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth = SmoothingFunction()
from tqdm import tqdm
import jieba
import numpy as np
def is_in_annotation(src, pos):
    s = 0
    count_left = 0
    while s < pos:
        if src[s] == '（':
            count_left += 1
        elif src[s] == '）':
            count_left -= 1
        s += 1
    if count_left > 0:
        return True
    else:
        return False
def fix_stop(tar):
    while re.search(r'（.*(。).*）', tar) is not None:
        tar_stop_list = re.finditer(r'（.*(。).*）', tar)
        for stop in tar_stop_list:
            if is_in_annotation(tar, stop.regs[1][0]):
                temp = list(tar)
                temp[stop.regs[1][0]] = '\\'
                tar = ''.join(temp)
            else:
                temp = list(tar)
                temp[stop.regs[1][0]] = '\n'
                tar = ''.join(temp)
    tar = tar.replace('\\', '')
    tar = tar.replace('\n', '。')
    return tar
def creat_sentence(src, tar):
    src = re.sub('\*\*', '', src)
    src = src.replace('(', '（')
    src = src.replace('$', '')
    src = src.replace(')', '）')
    src = src.replace('\n', '').replace('。。', '。')
    src = fix_stop(src)
    tar = re.sub('\*\*', '', tar)
    tar = tar.replace('\n', '').replace('。。', '。')
    tar = tar.replace('(', '（')
    tar = tar.replace(')', '）')
    tar = tar.replace('$', '')
    tar = fix_stop(tar)
    if src[-1] == '。' and tar[-1] != '。':
        tar += '。'
    if tar[-1] == '。' and src[-1] != '。':
        src += '。'
    data_key = None
    src_sts = src.split('。')
    tar_sts = tar.split('。')
    return src_sts, tar_sts