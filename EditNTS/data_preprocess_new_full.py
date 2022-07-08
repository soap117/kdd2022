import os
import pickle
import jieba
import nltk
import time
import json
import numpy as np
import requests
import pandas as pd
from rank_bm25 import BM25Okapi
import data
from tqdm import tqdm
from label_edits import sent2edit
import jieba.posseg as posseg
import torch
from config import config
from transformers import BertTokenizer
from cbert.modeling_cbert import BertForTokenClassification
# This script contains the reimplementation of the pre-process steps of the dataset
# For the editNTS system to run, the dataset need to be in a pandas DataFrame format
# with columns ['comp_tokens', 'simp_tokens','comp_ids','simp_ids', 'comp_pos_tags', 'comp_pos_ids', edit_labels','new_edit_ids']

PAD = 'PAD' #  This has a vocab id, which is used to represent out-of-vocabulary words [0]
UNK = 'UNK' #  This has a vocab id, which is used to represent out-of-vocabulary words [1]
KEEP = 'KEEP' # This has a vocab id, which is used for copying from the source [2]
DEL = 'DEL' # This has a vocab id, which is used for deleting the corresponding word [3]
START = 'START' # this has a vocab id, which is uded for indicating start of the sentence for decoding [4]
STOP = 'STOP' # This has a vocab id, which is used to stop decoding [5]
import re

def read_clean_data(path):
    sample_data = pickle.load(open(path, 'rb'))
    titles = []
    sections = []
    title2sections = {}
    urls = set()
    sec2id = {}
    for one in sample_data:
        if len(one['urls']) > 0:
            for tid, (title, url, tref) in enumerate(zip(one['rpsecs'], one['urls'], one['rsecs'])):
                if len(title) > 0:
                    web_title = title[-1]
                    web_title = re.sub('_.+', '', web_title)
                    web_title = re.sub(' -.+', '', web_title)
                    one['rpsecs'][tid][-1] = web_title
                    sections += title[0:-1]
                    titles.append(web_title)
                    if web_title in title2sections and url not in urls:
                        title2sections[web_title] += title[0:-1]
                        if tref not in title2sections[web_title]:
                            title2sections[web_title].append(tref)
                        urls.add(url)
                    elif web_title not in title2sections:
                        title2sections[web_title] = title[0:-1]
                        if tref not in title2sections[web_title]:
                            title2sections[web_title].append(tref)
                        urls.add(url)

    titles = list(set(titles))
    sections = list(set(sections))
    for k in range(len(sections)-1, -1, -1):
        if len(sections[k]) < 5:
            del sections[k]
    for tid, temp in enumerate(sections):
        sec2id[temp] = tid
    return titles, sections, title2sections, sec2id

titles, sections, title2sections, sec2id = read_clean_data(config.data_file_anno)
corpus = sections
tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
bm25_section = BM25Okapi(tokenized_corpus)
corpus = titles
tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
bm25_title = BM25Okapi(tokenized_corpus)

with open('../data/mydata_new_clean_v3_mark.pkl', 'rb') as f:
    key_tran = pickle.load(f)

save_step1_data = torch.load('./cbert/cache/' + 'best_save.data')
bert_model = 'hfl/chinese-bert-wwm-ext'
model_step1 = BertForTokenClassification.from_pretrained(bert_model, num_labels=4)
model_step1.load_state_dict(save_step1_data['para'])
model_step1.eval()
step1_tokenizer = BertTokenizer.from_pretrained(bert_model)
step1_tokenizer.model_max_length = 512

def remove_lrb(sent_string):
    # sent_string = sent_string.lower()
    frac_list = sent_string.split('-lrb-')
    clean_list = []
    for phrase in frac_list:
        if '-rrb-' in phrase:
            clean_list.append(phrase.split('-rrb-')[1].strip())
        else:
            clean_list.append(phrase.strip())
    clean_sent_string =' '.join(clean_list)
    return clean_sent_string

def replace_lrb(sent_string):
    sent_string = sent_string.lower()
    # new_sent= sent_string.replace('-lrb-','(').replace('-rrb-',')')
    new_sent = sent_string.replace('-lrb-', '').replace('-rrb-', '')
    return new_sent
def is_in_annotation(pos, src):
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
def obtain_annotation(tar , t_s):
    t_e = t_s + 1
    count = 1
    while t_e < len(tar) and count > 0:
        if tar[t_e] == '）':
            count -= 1
        if tar[t_e] == '（':
            count += 1
        t_e += 1
    annotations = tar[t_s+1:t_e-1]
    return annotations

def obtain_annotations(tar):
    t_s = 0
    annotations = []
    while t_s < len(tar):
        if tar[t_s] == '（':
            t_e = t_s + 1
            count = 1
            while t_e < len(tar) and count > 0:
                if tar[t_e] == '）':
                    count -= 1
                if tar[t_e] == '（':
                    count += 1
                t_e += 1
            anno_posi = tar[t_s:t_e]
            annotations.append(anno_posi)
            t_s = t_e
        else:
            t_s += 1
    return annotations

def modify_sentence(srcs, tars, query_groups):
    new_srcs = []
    new_tars = []
    new_query_groups = []
    for src, tar, query_group in zip(srcs, tars, query_groups):
        used = set()
        new_query_group = []
        for query_data in query_group:
            query = query_data[0]
            query = query.replace('(', '（')
            query = query.replace(')', '）')
            query = query.replace('（', '')
            query = query.replace('）', '')
            if query not in src:
                continue
            else:
                new_query_group.append(query_data)
            flag = False
            for key_used in used:
                if query in key_used:
                    flag = True
                    break
            if flag:
                continue
            regions = [x for x in re.finditer(query, tar)]
            region = None
            for one in regions:
                if is_in_annotation(one.regs[0][0], tar):
                    continue
                region = one.regs[0]
                break
            if region is None and len(regions) == 1:
                region = regions[0].regs[0]
            if region is not None:
                region = region
            else:
                region = (0, 0)
            if region[0] != 0 or region[1] != 0:
                if region[1] < len(tar) and tar[region[1]] == '（':
                    annotation = obtain_annotation(tar, region[1])
                    if annotation not in src:
                        tar = tar[0:region[0]] + '${}$'.format(query) + tar[region[1]:]
                        region = re.search(query, src)
                        if region is not None:
                            region = region.regs[0]
                        else:
                            region = (0, 0)
                        if region[0] != 0 or region[1] != 0:
                            src = src[0:region[0]] + '${}$'.format(query) + '（' + ''.join(
                                [' [MASK] ' for x in range(0)]) + '）' + src[region[1]:]
        new_query_groups.append(new_query_group)
        new_srcs.append(src)
        new_tars.append(tar)
    return new_srcs, new_tars, new_query_groups


def modify_sentence_direct(srcs, tars, query_groups):
    new_srcs = []
    new_tars = []
    new_query_groups = []
    for src, tar, query_group in zip(srcs, tars, query_groups):
        used = set()
        new_query_group = []
        for query_data in query_group:
            query = query_data[0]
            query = query.replace('(', '（')
            query = query.replace(')', '）')
            query = query.replace('（', '')
            query = query.replace('）', '')
            if query not in src:
                continue
            else:
                new_query_group.append(query_data)
            flag = False
            for key_used in used:
                if query in key_used:
                    flag = True
                    break
            if flag:
                continue
            regions = [x for x in re.finditer(query, tar)]
            region = None
            for one in regions:
                if is_in_annotation(one.regs[0][0], tar):
                    continue
                region = one.regs[0]
                break
            if region is None and len(regions) == 1:
                region = regions[0].regs[0]
            if region is not None:
                region = region
            else:
                region = (0, 0)
            if region[0] != 0 or region[1] != 0:
                if region[1] < len(tar):
                    tar = tar[0:region[0]] + '${}$'.format(query) + tar[region[1]:]

            region = re.search(query, src)
            if region is not None:
                region = region.regs[0]
            else:
                region = (0, 0)
            if region[0] != 0 or region[1] != 0:
                src = src[0:region[0]] + '${}$'.format(query) + '（' + ''.join(
                    [' [MASK] ' for x in range(0)]) + '）' + src[region[1]:]
        new_query_groups.append(new_query_group)
        new_srcs.append(src)
        new_tars.append(tar)
    return new_srcs, new_tars, new_query_groups



def obtain_step2_input(pre_labels, src, src_ids, step1_tokenizer):
    input_list = [[],[],[],[],[]]
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
            if len(key) > 0:
                input_list[0].append(key)
                input_list[1].append(context)
    return input_list

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

def modify_sentence_test(srcs, tars):
    new_srcs = []
    new_tars = []
    query_groups = []
    for src, tar in tqdm(zip(srcs, tars)):
        src_ = step1_tokenizer([src], return_tensors="pt", padding=True, truncation=True)
        x_ids = src_['input_ids']
        x_mask = src_['attention_mask']
        x_indicator = torch.zeros_like(x_ids)
        outputs = model_step1(x_ids, attention_mask=x_mask, existing_indicates=x_indicator)
        logits = outputs.logits
        pre_label_0 = np.argmax(logits.detach().cpu().numpy(), axis=2)
        x_indicator = obtain_indicator(x_ids[0], pre_label_0[0])
        x_indicator = torch.LongTensor(x_indicator).unsqueeze(0)
        outputs = model_step1(x_ids, attention_mask=x_mask, existing_indicates=x_indicator)
        logits = outputs.logits
        pre_label_f = np.argmax(logits.detach().cpu().numpy(), axis=2)
        step2_input = obtain_step2_input(pre_label_f[0], src, x_ids[0], step1_tokenizer)
        new_srcs.append(src)
        new_tars.append(tar)
        query_group = []
        for rs in zip(step2_input[0], step2_input[1]):
            query_group.append((rs[0], rs[1], None))
        query_groups.append(query_group)
    return new_srcs, new_tars, query_groups

def process_raw_data(train_data, is_train):
    # querys, querys_ori, querys_context, infer_titles, src_sens, src_sens_ori, tar_sens, edit_sens
    comp_txt = []
    simp_txt = []
    querys_group = []
    for comp, simp, query_gropu in train_data:
        comp_txt.append(comp)
        simp_txt.append(simp)
        querys_group.append(query_gropu)
    if is_train:
        comp_txt, simp_txt, querys_group = modify_sentence_direct(comp_txt, simp_txt, querys_group)
    else:
        comp_txt, simp_txt, querys_group = modify_sentence_test(comp_txt, simp_txt)
        comp_txt, simp_txt, _ = modify_sentence_direct(comp_txt, simp_txt, querys_group)
    comp_txt_pos = []
    for line in tqdm(comp_txt):
        comp_txt_pos.append(list(posseg.cut(line)))
    simp_txt_pos = []
    for line in tqdm(simp_txt):
        simp_txt_pos.append(list(posseg.cut(line)))
    comp_txt = [[x.word for x in line] for line in comp_txt_pos]
    simp_txt = [[x.word for x in line] for line in simp_txt_pos]
    # df_comp = pd.read_csv('mydata/%s_comp.csv'%dataset,  sep='\t')
    # df_simp= pd.read_csv('mydata/%s_simp.csv'%dataset,  sep='\t')
    assert len(comp_txt) == len(simp_txt)
    df = pd.DataFrame(
                        {'comp_tokens': comp_txt,
                         'simp_tokens': simp_txt,
                        })

    def add_querys(df):
        querys = []
        querys_ori = []
        querys_context = []
        infer_titles = []
        for query_group in querys_group:
            querys_sec = []
            querys_ori_sec = []
            querys_context_sec = []
            infer_titles_sec = []
            for query_set in query_group:
                query_ori = query_set[0]
                if query_ori in key_tran:
                    query_new = key_tran[query_ori]
                else:
                    url = 'https://api.ownthink.com/kg/ambiguous?mention=%s' % query_ori
                    headers = {
                        'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
                    }
                    try_count = 0
                    while try_count < 3:
                        try:
                            r = requests.get(url, headers=headers, timeout=5)
                            break
                        except Exception as e:
                            try_count += 1
                            print("trying %d time" % try_count)
                            wait_gap = 3
                            time.sleep((try_count + np.random.rand()) * wait_gap)
                    rs = json.loads(r.text)
                    rs = rs['data']
                    name_set = set()
                    name_set.add(query_ori)
                    for one in rs:
                        name_set.add(re.sub(r'\[.*\]', '', one[0]))
                    name_list = list(name_set)
                    query_new = ' '.join(name_list)
                    if len(query_new) == 0:
                        query_new = query_ori
                    if query_ori != query_new:
                        print("%s->%s" % (query_ori, query_new))
                    key_tran[query_ori] = query_new

                query_context = query_set[1]
                key_cut = jieba.lcut(query_new)
                infer_title = bm25_title.get_top_n(key_cut, titles, config.infer_title_range)

                querys_sec.append(query_new)
                querys_ori_sec.append(query_ori)
                querys_context_sec.append(query_context)
                infer_titles_sec.append(infer_title)

            querys.append(querys_sec)
            querys_ori.append(querys_ori_sec)
            querys_context.append(querys_context_sec)
            infer_titles.append(infer_titles_sec)

        df['querys'] = querys
        df['querys_ori'] = querys_ori
        df['query_contxt'] = querys_context
        df['infer_titles'] = infer_titles
        return df

    def get_vocab(df):
        word_dict ={}
        comp_sentences = df['comp_tokens'].tolist()
        simp_sentences = df['simp_tokens'].tolist()
        for sen in comp_sentences:
            for word in sen:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
        for sen in simp_sentences:
            for word in sen:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
        list_word = word_dict.items()
        list_word = sorted(list_word, key=lambda x: x[1], reverse=True)
        list_word = [x[0] +' '+ str(x[1]) + '\n' for x in list_word]
        f = open('./vocab_data/vocab.txt', 'w', encoding='utf-8')
        f.writelines(list_word)
        f.close()
    def add_edits(df):
        """
        :param df: a Dataframe at least contains columns of ['comp_tokens', 'simp_tokens']
        :return: df: a df with an extra column of target edit operations
        """
        comp_sentences = df['comp_tokens'].tolist()
        simp_sentences = df['simp_tokens'].tolist()
        pair_sentences = list(zip(comp_sentences,simp_sentences))
        edits_list = []
        for l in tqdm(pair_sentences):
            edits_list.append(sent2edit(l[0],l[1]))
        df['edit_labels'] = edits_list
        return df
    def create_pos_tag_table(pos_sentences):
        pos_tag_dict = {'PAD':0, 'UNK':1, 'START':2, 'STOP':3}
        for sent in pos_sentences:
            for word in sent:
                if word.flag not in pos_tag_dict:
                    pos_tag_dict[word.flag] = len(pos_tag_dict)
        with open('./vocab_data/chn_postag_set.p', 'wb') as f:
            pickle.dump(pos_tag_dict, f)
        return pos_tag_dict
    def add_pos(df, pos_sentences):
        src_sentences = df['comp_tokens'].tolist()
        df['comp_pos_tags'] = pos_sentences
        pos_vocab = data.POSvocab('./vocab_data/')
        pos_ids_list = []
        for sent in pos_sentences:
            pos_ids = [pos_vocab.w2i[w.flag] if w.flag in pos_vocab.w2i.keys() else pos_vocab.w2i[UNK] for w in sent]
            pos_ids_list.append(pos_ids)
        df['comp_pos_ids'] = pos_ids_list
        return df
    if is_train:
        get_vocab(df)
        create_pos_tag_table(comp_txt_pos)
    df = add_pos(df, comp_txt_pos)
    df = add_querys(df)
    df = add_edits(df)
    return df

def editnet_data_to_editnetID(df,output_path):
    """
    this function reads from df.columns=['comp_tokens', 'simp_tokens', 'edit_labels','comp_pos_tags','comp_pos_ids']
    and add vocab ids for comp_tokens, simp_tokens, and edit_labels
    :param df: df.columns=['comp_tokens', 'simp_tokens', 'edit_labels','comp_pos_tags','comp_pos_ids']
    :param output_path: the path to store the df
    :return: a dataframe with df.columns=['comp_tokens', 'simp_tokens', 'edit_labels',
                                            'comp_ids','simp_id','edit_ids',
                                            'comp_pos_tags','comp_pos_ids'])
    """
    out_list = []
    vocab = data.Vocab()
    vocab.add_vocab_from_file('./vocab_data/vocab.txt', 30000)

    def prepare_example(example, vocab):
        """
        :param example: one row in pandas dataframe with feild ['comp_tokens', 'simp_tokens', 'edit_labels']
        :param vocab: vocab object for translation
        :return: inp: original input sentence,
        """
        comp_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in example['comp_tokens']])
        simp_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in example['simp_tokens']])
        edit_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in example['edit_labels']])
        return comp_id, simp_id, edit_id  # add a dimension for batch, batch_size =1

    for i,example in df.iterrows():
        print(i)
        comp_id, simp_id, edit_id = prepare_example(example,vocab)
        ex=[example['comp_tokens'], comp_id,
         example['simp_tokens'], simp_id,
         example['edit_labels'], edit_id,
         example['comp_pos_tags'],example['comp_pos_ids']
         ]
        out_list.append(ex)
    outdf = pd.DataFrame(out_list, columns=['comp_tokens','comp_ids', 'simp_tokens','simp_ids',
                                            'edit_labels','new_edit_ids','comp_pos_tags','comp_pos_ids'])
    outdf.to_pickle(output_path)
    print('saved to %s'%output_path)


train_data = pickle.load(open('../data/train/dataset-aligned-para-new.pkl', 'rb'))
df = process_raw_data(train_data, True)
editnet_data_to_editnetID(df, './mydata/train_full.df.filtered.pos')
val_data = pickle.load(open('../data/valid/dataset-aligned-para-new.pkl', 'rb'))
df = process_raw_data(val_data, False)
editnet_data_to_editnetID(df, './mydata/val_full.df.filtered.pos')
test_data = pickle.load(open('../data/test/dataset-aligned-para-new.pkl', 'rb'))
df = process_raw_data(test_data, False)
editnet_data_to_editnetID(df, './mydata/test_full.df.filtered.pos')