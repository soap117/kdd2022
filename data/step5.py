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
        src_sts = src.split('。')
        tar_sts = tar.split('。')
        for i in range(len(src_sts)-1, -1, -1):
            if len(src_sts[i]) == 0:
                del src_sts[i]
        for i in range(len(tar_sts)-1, -1, -1):
            if len(tar_sts[i]) == 0:
                del tar_sts[i]
        dt = len(tar_sts) - len(src_sts)
        if dt == 0:
            for src_st, tar_st in zip(src_sts, tar_sts):
                if file['original_key'] in src_st and src_st != tar_st:
                    file['src_st'] = src_st
                    file['tar_st'] = tar_st
                    src_st_cut = list(src_st)
                    tar_st_cut = list(tar_st)
                    inversed_punishment = 1 / np.exp(1 - max(len(tar_st_cut), len(src_st_cut)) / min(len(tar_st_cut), len(src_st_cut)))
                    can_simi = sentence_bleu([tar_st_cut], src_st_cut)*inversed_punishment
                    if can_simi < 0.2:
                        tar_gt_st = None
                        src_gt_st = None
                        for src_st in src_sts:
                            if file['original_key'] in src_st:
                                src_gt_st = src_st
                                src_st_cut = list(src_st)
                                for tar_st in tar_sts:
                                    tar_st_cut = list(tar_st)
                                    inversed_punishment = 1 / np.exp(
                                        1 - max(len(tar_st_cut), len(src_st_cut)) / min(len(tar_st_cut),
                                                                                        len(src_st_cut)))
                                    can_simi = sentence_bleu([tar_st_cut], src_st_cut) * inversed_punishment
                                    if can_simi >= 0.5:
                                        tar_gt_st = tar_st
                                        break
                                break
                        if tar_gt_st is not None and src_gt_st != tar_gt_st:
                            file['src_st'] = src_gt_st
                            file['tar_st'] = tar_gt_st
                            pos = re.search(file['original_key'], src_st)
                            if pos is not None:
                                file['position'] = pos.regs
                                data_key = {'key': file['key'], 'origin': file['origin_key'], 'anno': file['anno'],
                                            'urls': file['urls'], 'rsecs': file['rsecs'],
                                            'rpsecs': file['rpsecs'], 'pos': file['position']}
                        else:
                            failed += 1
                            print('miss')
                    else:
                        pos = re.search(file['original_key'], src_st)
                        if pos is not None:
                            file['position'] = pos.regs
                            data_key = {'key': file['key'], 'origin': file['origin_key'], 'anno': file['anno'], 'urls': file['urls'], 'rsecs': file['rsecs'],
                                        'rpsecs': file['rpsecs'], 'pos': file['position']}
                    break
        else:
            tar_gt_st = None
            src_gt_st = None
            for src_st in src_sts:
                if file['original_key'] in src_st:
                    src_gt_st = src_st
                    src_st_cut = list(src_st)
                    for tar_st in tar_sts:
                        tar_st_cut = list(tar_st)
                        inversed_punishment = 1 / np.exp(
                            1 - max(len(tar_st_cut), len(src_st_cut)) / min(len(tar_st_cut), len(src_st_cut)))
                        can_simi = sentence_bleu([tar_st_cut], src_st_cut) * inversed_punishment
                        if can_simi >= 0.5:
                            tar_gt_st = tar_st
                            break
                    break
            if tar_gt_st is not None and src_gt_st != tar_gt_st:
                file['src_st'] = src_gt_st
                file['tar_st'] = tar_gt_st
                pos = re.search(file['original_key'], src_st)
                if pos is not None:
                    file['position'] = pos.regs
                    data_key = {'key': file['key'], 'origin': file['origin_key'], 'anno': file['anno'], 'urls': file['urls'], 'rsecs': file['rsecs'],
                                'rpsecs': file['rpsecs'], 'pos': file['position']}
            else:
                failed += 1
                print('miss')

        if data_key is not None:
            file_sen = file['file']['textid'] + file['src_st']
            if file_sen in sentence_format:
                sentence_format[file_sen]['data'].append(data_key)
            else:
                sentence_format[file_sen] = {}
                sentence_format[file_sen]['data'] = [data_key]
                sentence_format[file_sen]['src_st'] = file['src_st']
                sentence_format[file_sen]['tar_st'] = file['tar_st']
                sentence_format[file_sen]['textid'] = file['file']['textid']
    return list(sentence_format.values()), failed
with open('mydata_v5_anno.pkl', 'rb') as f:
    data_new = pickle.load(f)
sentence_format, failed = creat_sentence(data_new)
print(len(sentence_format))
print(failed)
with open('mydata_v5.pkl', 'wb') as f:
    pickle.dump(sentence_format, f)