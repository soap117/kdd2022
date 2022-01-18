import torch
import numpy as np
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth = SmoothingFunction()
def check(key, gtitles, candidata_title):
    key_cut = jieba.lcut(key)
    candidata_title = jieba.lcut(candidata_title)
    gtitles = [jieba.lcut(x) for x in gtitles]
    can_simi = sentence_bleu([candidata_title], key_cut, weights=(0.5, 0.5), smoothing_function=smooth.method1)
    grd_simi = sentence_bleu(gtitles, key_cut, weights=(0.5, 0.5), smoothing_function=smooth.method1)
    return can_simi < 0.25*grd_simi

def check_section(key, gsections, candidata_section, bm25, cind):
    key_cut = jieba.lcut(key)
    candidata_section = jieba.lcut(candidata_section)
    gsections = [jieba.lcut(x) for x in gsections]
    inversed_punishment = 1/np.exp(1-len(gsections[0])/len(key_cut))
    can_simi = sentence_bleu([candidata_section], key_cut, weights=(0.5, 0.5), smoothing_function=smooth.method1)*inversed_punishment
    can_simi_2 = sentence_bleu(gsections, candidata_section, weights=(0.5, 0.5), smoothing_function=smooth.method1)
    grd_simi = sentence_bleu(gsections, key_cut, weights=(0.5, 0.5), smoothing_function=smooth.method1)*inversed_punishment
    bm25_score = bm25.get_batch_scores(key_cut, [cind])
    return can_simi < 0.25*grd_simi and can_simi_2 < 0.25

def neg_sample_title(key, gtitles, candidate_titles, n):
    count = 0
    try_t = 0
    rs = []
    while count < n and try_t < 1000:
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
    while count < n and try_t < 1000:
        ind = np.random.randint(0, len(candidate_sections))
        candidate_section = candidate_sections[ind]
        if len(candidate_section) < 60:
            continue
        if check_section(key, gsections, candidate_section, bm25, ind):
            rs.append(candidate_section)
            count += 1
        try_t += 1
    return rs

def get_decoder_att_map(tokenizer, sep, ids, scores):
    spe_seq = tokenizer.encode(sep)
    mapping = np.zeros([len(ids), scores.shape[1], ids.shape[1]])
    for bindex, (bscore, bids) in enumerate(zip(scores, ids)):
        i = 0
        c_father = 0
        while i < len(bids):
            mapping[bindex, c_father, i] = 1
            i += 1
            k = 0
            while k<len(spe_seq) and i<len(bids) and (spe_seq[k]==bids[i]):
                i += 1
                k += 1
            if k == len(spe_seq):
                c_father += 1
    mapping = torch.FloatTensor(mapping).to(config.device)
    scores = scores.unsqueeze(1)
    scores = scores.matmul(mapping).squeeze(1)
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
    sample_title_candidates = []
    sample_section_candidates = []
    infer_title_candidates = []
    for key in keys:
        if len(key['rpsecs'][0]) <= 0:
            continue
        print('****************************************************')
        print(key['key'])
        sample_query.append(key['key'])
        key_cut = jieba.lcut(key['key'])
        infer_titles = bm25_title.get_top_n(key_cut, titles, 4)
        infer_title_candidates.append(infer_titles)
        neg_titles = neg_sample_title(key['key'], [x[-1] for x in key['rpsecs']], titles, 5)
        neg_sections = neg_sample_section(key['key'], key['rsecs'], sections, 5, bm25_section)
        pos_section = key['rsecs'][np.random.randint(len(key['rsecs']))]
        pos_title = key['rpsecs'][np.random.randint(len(key['rpsecs']))][-1]
        sample_title_candidates.append([pos_title] + neg_titles)
        sample_section_candidates.append([pos_section] + neg_sections)
        print('____________________________________________________')
        print(neg_titles)
        print('____________________________________________________')
        for x in neg_sections:
            print(x)
            print('____________________________________________________')
    return sample_query, sample_title_candidates, sample_section_candidates, infer_title_candidates

from torch.utils.data import Dataset, DataLoader

class MyData(Dataset):
    def __init__(self, config):
        if config.use_simple_reason:
            ans_length = 4
        else:
            ans_length = 11
        self.source = []
        self.target = []
        self.config = config
        self.data_path = config.data_path
