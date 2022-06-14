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
def creat_sentence(data_new):
    failed = 0
    sentence_format = {}
    for file in data_new:
        anno = file['file']
        src = anno['src']
        src = re.sub('\*\*', '', src)
        src = src.replace('(', '（')
        src = src.replace('$', '')
        src = src.replace(')', '）')
        src = src.replace('\n', '').replace('。。', '。')
        src = fix_stop(src)
        tar = anno['tar']
        tar = re.sub('\*\*', '', tar)
        tar = tar.replace('\n', '').replace('。。', '。')
        tar = tar.replace('(', '（')
        tar = tar.replace(')', '）')
        tar = tar.replace('$', '')
        tar = fix_stop(tar)
        file['origin_key'] = file['origin_key'].replace('(', '（')
        file['origin_key'] = file['origin_key'].replace(')', '）')
        file['key'] = file['key'].replace('(', '（')
        file['key'] = file['key'].replace(')', '）')
        if src[-1] == '。' and tar[-1] != '。':
            tar += '。'
        if tar[-1] == '。' and src[-1] != '。':
            src += '。'
        data_key = None
        if file['original_key'] in src and src != tar:
            pos = re.search(file['original_key'], src)
            if pos is not None:
                file['position'] = pos.regs
                data_key = {'key': file['key'], 'origin': file['origin_key'], 'anno': file['anno'],
                            'urls': file['urls'], 'rsecs': file['rsecs'],
                            'rpsecs': file['rpsecs'], 'pos': file['position']}


        if data_key is not None:
            file_sen = file['file']['textid'] + src
            if file_sen in sentence_format:
                sentence_format[file_sen]['data'].append(data_key)
            else:
                sentence_format[file_sen] = {}
                sentence_format[file_sen]['data'] = [data_key]
                sentence_format[file_sen]['src_st'] = src[0:512]
                sentence_format[file_sen]['tar_st'] = tar[0:512]
                sentence_format[file_sen]['textid'] = file['file']['textid']
        else:
            failed += 1
    return list(sentence_format.values()), failed
with open('mydata_v5_anno.pkl', 'rb') as f:
    data_new = pickle.load(f)
sentence_format, failed = creat_sentence(data_new)
print(len(sentence_format))
print(failed)
with open('mydata_v5_para.pkl', 'wb') as f:
    pickle.dump(sentence_format, f)