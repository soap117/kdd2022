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