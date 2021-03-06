import torch
import numpy as np
import jieba
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth = SmoothingFunction()
from tqdm import tqdm
from config import config
def mask_ref(input_ids, tokenizer):
    mask = np.random.choice([True, False], size=input_ids.shape, p=[0.15, 0.85])
    replace = np.random.choice(np.arange(tokenizer.vocab_size), size=input_ids.shape)
    input_ids = input_ids.numpy()
    input_ids[mask] = replace[mask]
    return torch.LongTensor(input_ids)
def batch_pointer_decode(source, pointers):
    temp = []
    source = source.cpu().numpy()
    pointers = pointers.cpu().numpy()
    for p_list, s_list in zip(pointers, source):
        temp_c = []
        for p in p_list:
            temp_c.append(s_list[p])
    temp.append(temp_c)
    return temp
def pointer_generation(source, target):
    sind = 0
    header = np.zeros(len(target))
    for wid, word in enumerate(target):
        while sind < len(source) and source[sind] != word:
            sind += 1
        if sind < len(source):
            header[wid] = sind
        else:
            sind = 0
            header[wid] = 1
    return header

def batch_pointer_generation(sources, targets):
    headers = []
    for source, target in zip(sources, targets):
        header_one = pointer_generation(source, target)
        headers.append(header_one)
    headers = np.array(headers)
    headers = torch.LongTensor(headers).to(config.device)
    return headers

def check(key, gtitles, candidata_title):
    key_cut = jieba.lcut(key)
    candidata_title = jieba.lcut(candidata_title)
    gtitles = [jieba.lcut(x) for x in gtitles]
    can_simi = sentence_bleu([candidata_title], key_cut, weights=(0.5, 0.5), smoothing_function=smooth.method1)
    grd_simi = sentence_bleu(gtitles, key_cut, weights=(0.5, 0.5), smoothing_function=smooth.method1)
    return can_simi < 0.25*grd_simi or grd_simi <= 0

def check_section(key, gsections, candidata_section, bm25, cind):
    key_cut = jieba.lcut(key)
    candidata_section = jieba.lcut(candidata_section)
    gsections = [jieba.lcut(x) for x in gsections]
    inversed_punishment = 1/np.exp(1-len(gsections[0])/len(key_cut))
    can_simi = sentence_bleu([candidata_section], key_cut, weights=(0.5, 0.5), smoothing_function=smooth.method1)*inversed_punishment
    can_simi_2 = sentence_bleu(gsections, candidata_section, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
    grd_simi = sentence_bleu(gsections, key_cut, weights=(0.5, 0.5), smoothing_function=smooth.method1)*inversed_punishment
    return (can_simi < 0.25*grd_simi or grd_simi <= 0) and can_simi_2 < 0.25

def check_section_strong(gsections, candidata_section, bm25, cind):
    candidata_section = jieba.lcut(candidata_section)
    gsections = [jieba.lcut(x) for x in gsections]
    can_simi_2 = sentence_bleu(gsections, candidata_section, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
    return can_simi_2 < 0.5

def neg_sample_title(key, gtitles, candidate_titles, n):
    count = 0
    try_t = 0
    rs = []
    while count < n and try_t < 100:
        ind = np.random.randint(0, len(candidate_titles))
        candidate_title = candidate_titles[ind]
        if check(key, gtitles, candidate_title):
            rs.append(candidate_title)
            count += 1
        try_t += 1
    return rs

def neg_sample_section(key, gsections, candidate_sections, n, bm25):
    count = 0
    rs = []
    try_t = 0
    while count < n and try_t < 100:
        ind = np.random.randint(0, len(candidate_sections))
        candidate_section = candidate_sections[ind]
        if len(candidate_section) < 5:
            continue
        if check_section(key, gsections, candidate_section, bm25, ind):
            rs.append(candidate_section)
            count += 1
        try_t += 1
    while count < n:
        count += 1
        rs.append("                          ")
    return rs

def neg_sample_strong_section(key, gsections, candidate_sections, n, bm25):
    count = 0
    rs = []
    try_t = 0
    while count < n and try_t < 100:
        ind = np.random.randint(0, len(candidate_sections))
        candidate_section = candidate_sections[ind]
        if len(candidate_section) < 5:
            try_t += 1
            continue
        if check_section_strong(gsections, candidate_section, bm25, ind):
            rs.append(candidate_section)
            count += 1
        try_t += 1
    while count < n:
        count += 1
        rs.append("                          ")
    return rs

def get_decoder_att_map(tokenizer, sep, ids, scores):
    spe_seq = [config.SEP]
    mapping = np.zeros([len(ids), scores.shape[1], ids.shape[1]])
    adding = np.zeros([len(ids), ids.shape[1]])
    for bindex, (bscore, bids) in enumerate(zip(scores, ids)):
        i = 0
        c_father = -1
        while i < len(bids):
            if c_father >= 0 and c_father < scores.shape[1]:
                mapping[bindex, c_father, i] = 1
            else:
                adding[bindex, i] = 1
            i += 1
            k = 0
            while k<len(spe_seq) and i<len(bids) and (spe_seq[k]==bids[i]):
                #mapping[bindex, c_father, i] = 1
                adding[bindex, i] = 1
                i += 1
                k += 1
            if k == len(spe_seq):
                c_father += 1
    mapping = torch.FloatTensor(mapping).to(config.device)
    scores = scores.unsqueeze(1)
    scores = scores.matmul(mapping).squeeze(1) + torch.FloatTensor(adding).to(config.device)
    return scores

def get_sec_att_map(sample_query, input_inds, infer_title_candidates, title2sections, sec2id, bm25_section):
    infer_title_candidates_pured = []
    infer_section_candidates_pured = []
    mapping_title = np.zeros([len(sample_query), 3, 4])
    for query, bid in zip(sample_query, range(len(input_inds))):
        temp = []
        temp2 = []
        temp3 = []
        for nid, cid in enumerate(input_inds[bid]):
            temp.append(infer_title_candidates[bid][cid])
            temp2 += title2sections[infer_title_candidates[bid][cid]]
            temp3 += [nid for x in title2sections[infer_title_candidates[bid][cid]]]
        temp2_id = []
        for t_sec in temp2:
            if t_sec in sec2id:
                temp2_id.append(sec2id[t_sec])
        key_cut = jieba.lcut(query)
        ls_scores = bm25_section.get_batch_scores(key_cut, temp2_id)
        cindex = np.argsort(ls_scores)[::-1][0:4]
        temp2_pured = []
        temp3_pured = []
        for oid, one in enumerate(cindex):
            temp2_pured.append(temp2[one])
            temp3_pured.append(temp3[one])
            mapping_title[bid, temp3[one], oid] = 1
        while len(temp2_pured) < 4:
            temp2_pured.append('')
            temp3_pured.append(-1)

        infer_title_candidates_pured.append(temp)
        infer_section_candidates_pured.append(temp2_pured)
    return mapping_title, infer_title_candidates_pured, infer_section_candidates_pured

def get_retrieval_train_batch(keys, titles, sections, bm25_title, bm25_section):
    sample_query = []
    sample_annotation = []
    sample_title_candidates = []
    sample_section_candidates = []
    infer_title_candidates = []
    sample_pos_ans = []
    sampe_strong_candidates = []
    sample_query_context = []
    for key in tqdm(keys):
        if len(key['anno']) == 0:
            continue
        s = time.time()
        if len(key['rpsecs'][0]) <= 1 or len(key['key']) < 1:
            continue
        sample_query_context.append(key['sentence'])
        sample_query.append(key['key'])
        sample_annotation.append(key['anno'])
        key_cut = jieba.lcut(key['key'])
        infer_titles = bm25_title.get_top_n(key_cut, titles, config.infer_title_range)
        infer_title_candidates.append(infer_titles)
        for x in key['rpsecs']:
            if len(x) == 0:
                x.append('')
        neg_titles = neg_sample_title(key['key'], [x[-1] for x in key['rpsecs']], titles, config.neg_num)
        neg_sections = neg_sample_section(key['key'], key['rsecs'], sections, config.neg_num, bm25_section)
        temp_strong_neg_sections = []
        for _ in key['rpsecs']:
            temp_strong_neg_sections += _
        neg_sections_strong = neg_sample_strong_section(key['key'], key['rsecs'], temp_strong_neg_sections, config.neg_strong_num, bm25_section)
        if len(neg_sections_strong) <= 0:
            neg_sections_strong.append('                                                            ')
        #pos_section = key['rsecs'][np.random.randint(len(key['rsecs']))]
        #pos_title = key['rpsecs'][np.random.randint(len(key['rpsecs']))][-1]
        sample_pos_ans.append((key['rsecs'], key['rpsecs']))
        sample_title_candidates.append(neg_titles)
        sample_section_candidates.append(neg_sections)
        sampe_strong_candidates.append(neg_sections_strong)
        e = time.time()
        if e-s > 5:
            print(key['key'])
    return sample_query, sample_title_candidates, sample_section_candidates, infer_title_candidates, sample_pos_ans, sample_annotation, sampe_strong_candidates, sample_query_context

from torch.utils.data import Dataset
import pickle, re
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

def read_data(path):
    sample_data = pickle.load(open(path, 'rb'))
    keys = []
    for one in sample_data:
        if len(one['urls']) > 0:
            for tid, (title, url) in enumerate(zip(one['rpsecs'], one['urls'])):
                if len(title) > 0:
                    web_title = title[-1]
                    web_title = re.sub('_.+', '', web_title)
                    web_title = re.sub(' -.+', '', web_title)
                    one['rpsecs'][tid][-1] = web_title
            keys.append(one)
    return keys


class MyData(Dataset):
    def __init__(self, config, tokenizer, data_path, titles, sections, title2sections, sec2id, bm25_title, bm25_section):
        self.config = config
        keys = read_data(data_path)
        self.title2sections = title2sections
        self.sec2id = sec2id
        sample_query, sample_title_candidates, sample_section_candidates, infer_title_candidates, \
        sample_pos_ans, sample_annotation, sample_strong_section_candidates, sample_query_context \
            = get_retrieval_train_batch(keys, titles, sections, bm25_title, bm25_section)
        self.sample_query = sample_query
        self.sample_query_context = sample_query_context
        self.sample_title_candidates = sample_title_candidates
        self.sample_section_candidates = sample_section_candidates
        self.sample_strong_section_candidates = sample_strong_section_candidates
        self.sample_pos_ans = sample_pos_ans
        self.sample_annotation = sample_annotation
        self.infer_title_candidates = infer_title_candidates
        self.bm25_title = bm25_title
        self.bm25_section = bm25_section
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sample_query)

    def __getitem__(self, item):
        pos_section = self.sample_pos_ans[item][0][np.random.randint(len(self.sample_pos_ans[item][0]))]
        strong_neg_section = self.sample_strong_section_candidates[item][np.random.randint(len(self.sample_strong_section_candidates[item]))]
        pos_title = self.sample_pos_ans[item][1][np.random.randint(len(self.sample_pos_ans[item][1]))][-1]
        sample_query = self.sample_query[item]
        sample_title_candidates = [pos_title] + self.sample_title_candidates[item]
        sample_section_candidates = [pos_section] + self.sample_section_candidates[item] + [strong_neg_section]
        infer_title_candidates = self.infer_title_candidates[item]
        sample_annotation = self.sample_annotation[item]
        pos_titles = [x[-1] for x in self.sample_pos_ans[item][1]]
        pos_sections = self.sample_pos_ans[item][0]
        sample_query_context = self.sample_query_context[item]
        return sample_query, sample_title_candidates, sample_section_candidates, infer_title_candidates, sample_annotation, pos_titles, pos_sections, sample_query_context

    def collate_fn(self, train_data):
        querys = [data[0] for data in train_data]
        querys_context = [data[7] for data in train_data]
        titles = [data[1] for data in train_data]
        sections = [data[2] for data in train_data]
        infer_titles = [data[3] for data in train_data]
        annotations = [data[4] for data in train_data]
        return querys, querys_context, titles, sections, infer_titles, annotations

    def collate_fn_test(self, train_data):
        querys = [data[0] for data in train_data]
        querys_context = [data[7] for data in train_data]
        titles = [data[1] for data in train_data]
        sections = [data[2] for data in train_data]
        infer_titles = [data[3] for data in train_data]
        annotations = [data[4] for data in train_data]
        pos_titles = [data[5] for data in train_data]
        pos_sections = [data[6] for data in train_data]
        return querys, querys_context, titles, sections, infer_titles, annotations, pos_titles, pos_sections





