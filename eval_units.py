from config import config
from nltk.translate.bleu_score import sentence_bleu
from transformers import BertTokenizer
import jieba
from rank_bm25 import BM25Okapi
bert_model_eval = 'hfl/chinese-bert-wwm-ext'
from models.units import read_clean_data
import pickle
tokenzier_eval = BertTokenizer.from_pretrained(bert_model_eval)
titles, sections, title2sections, sec2id = read_clean_data(config.data_file_anno)
corpus = sections
tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
bm25_section = BM25Okapi(tokenized_corpus)
step2_tokenizer = config.tokenizer
step2_tokenizer.model_max_length = 300
corpus = titles
tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
bm25_title = BM25Okapi(tokenized_corpus)
with open('./data/mydata_new_clean_v3_mark.pkl', 'rb') as f:
    mark_key_equal = pickle.load(f)

def get_sentence_bleu(candidate, reference):
    score = sentence_bleu(reference, candidate)
    return score


def count_score(candidate, reference):
    avg_score = 0
    for k in range(len(candidate)):
        reference_ = reference[k]
        for m in range(len(reference_)):
            reference_[m] = tokenzier_eval.tokenize(reference_[m])
        candidate[k] = tokenzier_eval.tokenize(candidate[k])
        try:
            avg_score += get_sentence_bleu(candidate[k], reference_)/len(candidate)
        except:
            print(candidate[k])
            print(reference[k])
    return avg_score

def obtain_step2_input(pre_labels, src, src_ids, step1_tokenizer):
    input_list = [[],[],[], [],[]]
    l = 0
    r = 0
    while src_ids[r] != step1_tokenizer.vocab['。']:
        r += 1
    for c_id in range(len(src_ids)):
        if src_ids[c_id] == step1_tokenizer.vocab['。']:
            context = step1_tokenizer.decode(src_ids[l:r]).replace(' ', '').replace('[CLS]', '').replace('[SEP]', '')
            input_list[4].append((False, context))
            l = r + 1
            r = l+1
            while r<len(src_ids) and src_ids[r] != step1_tokenizer.vocab['。']:
                r += 1
        if pre_labels[c_id] == 1:
            l_k = c_id
            r_k = l_k+1
            while r_k<len(pre_labels) and pre_labels[r_k] == 3:
                r_k += 1
            templete = src_ids[l_k:r_k]
            tokens = step1_tokenizer.convert_ids_to_tokens(templete)
            key = step1_tokenizer.convert_tokens_to_string(tokens).replace(' ', '')
            context = step1_tokenizer.decode(src_ids[l:r]).replace(' ', '').replace('[CLS]', '').replace('[SEP]', '')
            key_cut = jieba.lcut(key)
            infer_titles = bm25_title.get_top_n(key_cut, titles, config.infer_title_range)
            if len(key) > 0:
                input_list[0].append(key)
                input_list[1].append(context)
                input_list[2].append(infer_titles)
                input_list[3].append((l_k, r_k))
    return input_list

def mark_sentence_rnn(input_list):
    context_dic = {}
    for key, context, infer_titles in zip(input_list[0], input_list[1], input_list[2]):
        if context not in context_dic:
            src = context
            src = re.sub('\*\*', '', src)
            src = src.replace('(', '（')
            src = src.replace('$', '')
            src = src.replace(')', '）')
            src = src.replace('\n', '').replace('。。', '。')
            src = fix_stop(src)
            context_dic[context] = [src, src, [], []]
            src_sentence = context_dic[context][0]
            tar_sentence = context_dic[context][1]

        else:
            src_sentence = context_dic[context][0]
            tar_sentence = context_dic[context][1]
        context_dic[context][2].append(key)
        context_dic[context][3].append(infer_titles)
        region = re.search(key, src_sentence)
        if region is not None:
            region = region.regs[0]
        else:
            region = (0, 0)
        if region[0] != 0 or region[1] != 0:
            src_sentence = src_sentence[0:region[0]] + ' ${}$ '.format(key) + '（' + ''.join(
                [' [unused3] ']+[' [MASK] ' for x in range(config.hidden_anno_len_rnn-2)] + [' [unused4] ']) + '）' + src_sentence[region[1]:]
        region = re.search(key, tar_sentence)
        if region is not None:
            region = region.regs[0]
        else:
            region = (0, 0)
        if region[0] != 0 or region[1] != 0:
            tar_sentence = tar_sentence[0:region[0]] + ' ${}$ （）'.format(key) + tar_sentence[region[1]:]

        context_dic[context][0] = src_sentence
        context_dic[context][1] = tar_sentence
    order_context = []
    for context in input_list[4]:
        if context[1] not in order_context:
            order_context.append(context[1])
    return context_dic, order_context

def mark_sentence_rnn_para(input_list, src):
    context_dic = {}
    src = re.sub('\*\*', '', src)
    src = src.replace('(', '（')
    src = src.replace('$', '')
    src = src.replace(')', '）')
    src = src.replace('\n', '').replace('。。', '。')
    src = fix_stop(src)
    for key, context, infer_titles in zip(input_list[0], input_list[1], input_list[2]):
        if src not in context_dic:
            context_dic[src] = [src, src, [], [], []]
            src_sentence = context_dic[src][0]
            tar_sentence = context_dic[src][1]

        else:
            src_sentence = context_dic[src][0]
            tar_sentence = context_dic[src][1]
        context_dic[src][2].append(key)
        context_dic[src][3].append(infer_titles)
        context_dic[src][4].append(context)
        region = re.search(key, src_sentence)
        if region is not None:
            region = region.regs[0]
        else:
            region = (0, 0)
        if region[0] != 0 or region[1] != 0:
            src_sentence = src_sentence[0:region[0]] + ' ${}$ '.format(key) + '（' + ''.join(
                [' [unused3] ']+[' [MASK] ' for x in range(config.hidden_anno_len_rnn-2)] + [' [unused4] ']) + '）' + src_sentence[region[1]:]
        region = re.search(key, tar_sentence)
        if region is not None:
            region = region.regs[0]
        else:
            region = (0, 0)
        if region[0] != 0 or region[1] != 0:
            tar_sentence = tar_sentence[0:region[0]] + ' ${}$ （）'.format(key) + tar_sentence[region[1]:]

        context_dic[src][0] = src_sentence
        context_dic[src][1] = tar_sentence
    order_context = []
    for context in input_list[4]:
        if context[1] not in order_context:
            order_context.append(context[1])
    return context_dic, order_context

def mark_sentence(input_list):
    context_dic = {}
    for key, context, infer_titles in zip(input_list[0], input_list[1], input_list[2]):
        if context not in context_dic:
            src = context
            src = re.sub('\*\*', '', src)
            src = src.replace('(', '（')
            src = src.replace('$', '')
            src = src.replace(')', '）')
            src = src.replace('\n', '').replace('。。', '。')
            src = fix_stop(src)
            context_dic[context] = [src, src, [], []]
            src_sentence = context_dic[context][0]
            tar_sentence = context_dic[context][1]

        else:
            src_sentence = context_dic[context][0]
            tar_sentence = context_dic[context][1]
        context_dic[context][2].append(key)
        context_dic[context][3].append(infer_titles)
        region = re.search(key, src_sentence)
        if region is not None:
            region = region.regs[0]
        else:
            region = (0, 0)
        if region[0] != 0 or region[1] != 0:
            src_sentence = src_sentence[0:region[0]] + ' ${}$ '.format(key) + '（' + ''.join(
                [' [unused3] ']+[' [MASK] ' for x in range(config.hidden_anno_len-2)] + [' [unused4] ']) + '）' + src_sentence[region[1]:]
        region = re.search(key, tar_sentence)
        if region is not None:
            region = region.regs[0]
        else:
            region = (0, 0)
        if region[0] != 0 or region[1] != 0:
            tar_sentence = tar_sentence[0:region[0]] + ' ${}$ （）'.format(key) + tar_sentence[region[1]:]

        context_dic[context][0] = src_sentence
        context_dic[context][1] = tar_sentence
    order_context = []
    for context in input_list[4]:
        if context[1] not in order_context:
            order_context.append(context[1])
    return context_dic, order_context

import re
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